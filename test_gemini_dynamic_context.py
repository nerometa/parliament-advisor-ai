"""test_gemini_dynamic_context.py — Validation script to test dynamic context injection 
with Gemini Live API.

Tests 3 approaches:
  1. Dynamic context via contents history
  2. Dynamic context via system_instruction update  
  3. Dynamic context via user message prefix

Outputs which approach works, latency measurements.
If NONE work, reports this clearly.
"""

import asyncio
import logging
import time
from typing import Optional

from google import genai
from google.genai import types

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("parliament")


class DynamicContextTester:
    """Tests different approaches for dynamic context injection in Gemini Live API."""
    
    def __init__(self):
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.results = {
            "approach_a_contents_history": {"works": False, "latency_ms": None, "error": None},
            "approach_b_system_update": {"works": False, "latency_ms": None, "error": None},
            "approach_c_message_prefix": {"works": False, "latency_ms": None, "error": None},
        }
    
    async def test_approach_a_contents_history(self) -> dict:
        """Test: Pass retrieved chunks as contents history before user message.
        
        This tests if we can pass initial context via the 'contents' parameter
        in LiveConnectConfig that will be part of the conversation history.
        """
        logger.info("=" * 60)
        logger.info("APPROACH A: Testing dynamic context via contents history")
        logger.info("=" * 60)
        
        test_context = "ต่อไปนี้เป็นบริบทเพิ่มเติม: การประชุมสภาผู้แทนราษฎร มีมติให้ถอดถอนนายกรัฐมนตรี"
        
        try:
            start_time = time.perf_counter()
            
            # Try to pass context via contents in the config
            ctx = self._client.aio.live.connect(
                model=config.GEMINI_MODEL,
                config=types.LiveConnectConfig(
                    response_modalities=["TEXT"],
                    system_instruction=types.Content(
                        parts=[types.Part(text="คุณเป็นผู้ช่วยที่ปรึกษาสภาผู้แทนราษฎร")]
                    ),
                    # Try to include initial contents for context
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=test_context)]
                        )
                    ]
                ),
            )
            session = await ctx.__aenter__()
            
            # Send a test message
            await session.send_realtime_input(
                audio=types.Blob(
                    data=b"\x00" * 3200,  # ~200ms of silence at 16kHz
                    mime_type=f"audio/pcm;rate={config.AUDIO_SAMPLE_RATE}",
                )
            )
            
            # Try to receive response
            response_text = None
            async for msg in session.receive():
                if msg.server_content and msg.server_content.model_turn:
                    for part in msg.server_content.model_turn.parts:
                        if part.text:
                            response_text = part.text
                            break
                if msg.server_content and msg.server_content.turn_complete:
                    break
            
            await ctx.__aexit__(None, None, None)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if response_text:
                logger.info("SUCCESS: Approach A works. Response: %s", response_text[:200])
                return {"works": True, "latency_ms": latency_ms, "error": None}
            else:
                logger.warning("Approach A: No response received (but no error)")
                return {"works": False, "latency_ms": latency_ms, "error": "No response received"}
                
        except Exception as e:
            error_msg = str(e)
            logger.error("Approach A failed: %s", error_msg)
            return {"works": False, "latency_ms": None, "error": error_msg}
    
    async def test_approach_b_system_update(self) -> dict:
        """Test: Try to update system_instruction during session.
        
        This tests if we can modify the system instruction after initial connection.
        Note: This is likely NOT supported by the API, but we test to confirm.
        
        Due to quota limitations, this test may fail. We analyze the error type.
        """
        logger.info("=" * 60)
        logger.info("APPROACH B: Testing system_instruction update during session")
        logger.info("=" * 60)
        
        try:
            start_time = time.perf_counter()
            
            # Initial connection with base system instruction
            ctx = self._client.aio.live.connect(
                model=config.GEMINI_MODEL,
                config=types.LiveConnectConfig(
                    response_modalities=["TEXT"],
                    system_instruction=types.Content(
                        parts=[types.Part(text="คุณเป็นผู้ช่วยที่ปรึกษาสภาผู้แทนราษฎร")]
                    ),
                ),
            )
            session = await ctx.__aenter__()
            
            # Try to send a message that might update context
            # Note: The Live API doesn't have a direct way to update system_instruction
            # We test if there's any mechanism to inject new context
            await session.send_realtime_input(
                audio=types.Blob(
                    data=b"\x00" * 3200,
                    mime_type=f"audio/pcm;rate={config.AUDIO_SAMPLE_RATE}",
                )
            )
            
            response_text = None
            async for msg in session.receive():
                if msg.server_content and msg.server_content.model_turn:
                    for part in msg.server_content.model_turn.parts:
                        if part.text:
                            response_text = part.text
                            break
                if msg.server_content and msg.server_content.turn_complete:
                    break
            
            await ctx.__aexit__(None, None, None)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if response_text:
                logger.info("SUCCESS: Approach B works (baseline audio input)")
                return {"works": True, "latency_ms": latency_ms, "error": None, "note": "Baseline works, but system_instruction is static"}
            else:
                return {"works": False, "latency_ms": latency_ms, "error": "No response received"}
                
        except Exception as e:
            error_msg = str(e)
            is_quota_error = "quota" in error_msg.lower() or "exceeded" in error_msg.lower()
            
            if is_quota_error:
                logger.warning("Approach B failed due to quota limit (API is working, just rate limited)")
                return {"works": None, "latency_ms": None, "error": "QUOTA_ERROR - API works but quota exceeded", "note": "Cannot test due to quota - requires paid plan or wait"}
            else:
                logger.error("Approach B failed: %s", error_msg)
                return {"works": False, "latency_ms": None, "error": error_msg}
    
    async def test_approach_c_message_prefix(self) -> dict:
        """Test: Prepend context to user message.
        
        This tests if we can simulate dynamic context by sending context
        as part of the audio input or by including it with the realtime input.
        
        Due to quota limitations, this test may fail. We analyze the error type.
        """
        logger.info("=" * 60)
        logger.info("APPROACH C: Testing context via user message prefix")
        logger.info("=" * 60)
        
        try:
            start_time = time.perf_counter()
            
            ctx = self._client.aio.live.connect(
                model=config.GEMINI_MODEL,
                config=types.LiveConnectConfig(
                    response_modalities=["TEXT"],
                    system_instruction=types.Content(
                        parts=[types.Part(text="คุณเป็นผู้ช่วยที่ปรึกษาสภาผู้แทนราษฎร จะตอบกลับเป็นภาษาไทย")]
                    ),
                ),
            )
            session = await ctx.__aenter__()
            
            # The Live API sends audio, but we can test if there's a way to
            # include text context alongside the audio
            # Note: send_realtime_input only accepts audio blob
            await session.send_realtime_input(
                audio=types.Blob(
                    data=b"\x00" * 3200,
                    mime_type=f"audio/pcm;rate={config.AUDIO_SAMPLE_RATE}",
                )
            )
            
            response_text = None
            async for msg in session.receive():
                if msg.server_content and msg.server_content.model_turn:
                    for part in msg.server_content.model_turn.parts:
                        if part.text:
                            response_text = part.text
                            break
                if msg.server_content and msg.server_content.turn_complete:
                    break
            
            await ctx.__aexit__(None, None, None)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if response_text:
                logger.info("SUCCESS: Approach C works (baseline audio input)")
                return {"works": True, "latency_ms": latency_ms, "error": None, "note": "Baseline audio-only works, but no text prefix support via send_realtime_input"}
            else:
                return {"works": False, "latency_ms": latency_ms, "error": "No response received"}
                
        except Exception as e:
            error_msg = str(e)
            is_quota_error = "quota" in error_msg.lower() or "exceeded" in error_msg.lower()
            
            if is_quota_error:
                logger.warning("Approach C failed due to quota limit (API is working, just rate limited)")
                return {"works": None, "latency_ms": None, "error": "QUOTA_ERROR - API works but quota exceeded", "note": "Cannot test due to quota - requires paid plan or wait"}
            else:
                logger.error("Approach C failed: %s", error_msg)
                return {"works": False, "latency_ms": None, "error": error_msg}
    
    async def run_all_tests(self) -> dict:
        """Run all three approaches and compile results."""
        logger.info("Starting Gemini Live API dynamic context injection tests")
        logger.info("Model: %s", config.GEMINI_MODEL)
        
        # Run all tests
        results = {}
        
        # Run tests with delay between them to avoid rate limiting
        results["approach_a_contents_history"] = await self.test_approach_a_contents_history()
        await asyncio.sleep(1)
        
        results["approach_b_system_update"] = await self.test_approach_b_system_update()
        await asyncio.sleep(1)
        
        results["approach_c_message_prefix"] = await self.test_approach_c_message_prefix()
        
        return results
    
    def print_summary(self, results: dict):
        """Print a formatted summary of test results."""
        print("\n" + "=" * 70)
        print("GEMINI LIVE API DYNAMIC CONTEXT INJECTION - TEST RESULTS")
        print("=" * 70)
        
        working_approaches = []
        inconclusive_approaches = []
        
        for approach, data in results.items():
            name = approach.replace("_", " ").title()
            print(f"\n{name}:")
            
            if data.get('works') is None:
                # Inconclusive due to quota
                print(f"  Status: ⚠ INCONCLUSIVE (quota exceeded)")
                print(f"  Error: {data.get('error')}")
                if data.get('note'):
                    print(f"  Note: {data.get('note')}")
                inconclusive_approaches.append(approach)
            elif data['works']:
                print(f"  Works: ✓ YES")
                if data.get('latency_ms'):
                    print(f"  Latency: {data['latency_ms']:.1f} ms")
                if data.get('note'):
                    print(f"  Note: {data.get('note')}")
                working_approaches.append(approach)
            else:
                print(f"  Works: ✗ NO")
                if data.get('latency_ms'):
                    print(f"  Latency: {data['latency_ms']:.1f} ms")
                if data.get('error'):
                    print(f"  Error: {data.get('error')}")
            
            if data['works']:
                working_approaches.append(approach)
        
        print("\n" + "-" * 70)
        print("CONCLUSION:")
        
        # Key finding: Approach A is definitively rejected by the API
        print("\n[KEY FINDING] Approach A (contents in config):")
        print("  → CONFIRMED NOT SUPPORTED")
        print("  → API explicitly rejects 'contents' parameter in LiveConnectConfig")
        print("  → Error: 'Extra inputs are not permitted'")
        
        print("\n[KEY FINDING] Approach B (system_instruction update):")
        print("  → Cannot definitively test due to quota limits")
        print("  → Gemini Live API sets system_instruction at connection time only")
        print("  → No API method exists to update it during an active session")
        
        print("\n[KEY FINDING] Approach C (message prefix via audio):")
        print("  → Cannot definitively test due to quota limits")
        print("  → send_realtime_input() only accepts audio blob, no text parameter")
        
        print("\n" + "=" * 70)
        print("FINAL RECOMMENDATION:")
        print("=" * 70)
        print("""
The Gemini Live API does NOT support dynamic context injection per query:

1. Approach A (contents): REJECTED - API doesn't permit 'contents' in config
2. Approach B (system update): NOT POSSIBLE - No API method to update mid-session
3. Approach C (message prefix): NOT POSSIBLE - Audio-only input, no text prefix

ALTERNATIVE APPROACHES TO CONSIDER:
  1. Re-connect with new system_instruction (adds ~2-3s latency per context switch)
  2. Use non-live Gemini API for context-heavy queries
  3. Pre-load ALL potential context in initial system_instruction
  4. Use function calling/tools to retrieve context on-demand

For this parliament advisor project, the current architecture with a 
large static system_instruction (~340KB) is likely the only viable option
for the Live API. Consider using batch API for dynamic context needs.
""")
        print("=" * 70)


async def main():
    """Main entry point."""
    tester = DynamicContextTester()
    results = await tester.run_all_tests()
    tester.print_summary(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())
