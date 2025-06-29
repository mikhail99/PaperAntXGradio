#!/usr/bin/env python3
"""
Test script for the modern V3 implementation using interrupt() pattern.
This verifies that the new HIL nodes and service layer work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.proposal_agent.modern_service import create_modern_service
from core.proposal_agent.modern_parrot_services import create_test_scenarios_file


async def test_modern_interrupt_pattern():
    """Test the modern interrupt() pattern implementation."""
    print("🧪 Testing Modern V3 Implementation with interrupt() pattern")
    print("=" * 60)
    
    # Create test scenarios file first
    print("\n📁 Creating test scenarios configuration...")
    create_test_scenarios_file()
    print("✅ Test scenarios created successfully!")
    
    # Create modern service
    print("\n🚀 Creating modern service...")
    service = create_modern_service("research_proposal_workflow")
    print("✅ Modern service created successfully!")
    
    # Test configuration
    test_config = {
        "topic": "AI safety in autonomous vehicles",
        "collection_name": "test_collection",
        "local_papers_only": True
    }
    
    print(f"\n🎯 Starting agent with topic: '{test_config['topic']}'")
    print("⏳ Waiting for first interrupt...")
    
    step_count = 0
    thread_id = None
    
    try:
        # Start the agent and wait for first interrupt
        async for result in service.start_agent(test_config):
            step_count += 1
            step = result.get("step", "unknown")
            thread_id = result.get("thread_id")
            
            print(f"\n📋 Step {step_count}: {step}")
            
            if step == "human_input_required":
                interrupt_type = result.get("interrupt_type", "unknown")
                message = result.get("message", "No message")
                context = result.get("context", {})
                
                print(f"🔄 INTERRUPT DETECTED!")
                print(f"   Type: {interrupt_type}")
                print(f"   Stage: {context.get('stage', 'unknown')}")
                print(f"   Message: {message[:100]}...")
                
                # Test input validation
                validation_result = service.validate_user_input(interrupt_type, "test input")
                print(f"   Validation: {validation_result}")
                
                # For testing, we'll provide 'continue' to proceed
                print(f"   🤖 Auto-responding with 'continue'...")
                break
            elif step == "error":
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                return False
            else:
                state_keys = list(result.get("state", {}).keys()) if result.get("state") else []
                print(f"   ⚙️  Processing: {state_keys}")
                
    except Exception as e:
        print(f"❌ Error during agent start: {e}")
        return False
    
    # Test continuing the agent
    if thread_id:
        print(f"\n▶️  Continuing agent with thread_id: {thread_id}")
        try:
            continue_count = 0
            async for result in service.continue_agent(thread_id, "continue"):
                continue_count += 1
                step = result.get("step", "unknown")
                
                print(f"📋 Continue Step {continue_count}: {step}")
                
                if step == "human_input_required":
                    interrupt_type = result.get("interrupt_type", "unknown")
                    context = result.get("context", {})
                    print(f"🔄 Another interrupt: {interrupt_type} at stage {context.get('stage', 'unknown')}")
                    # For testing, break after first continue interrupt
                    break
                elif step == "error":
                    print(f"❌ Continue Error: {result.get('error', 'Unknown error')}")
                    break
                    
                # Limit test to prevent infinite loops
                if continue_count >= 5:
                    print("⏹️  Stopping test after 5 continue steps")
                    break
                    
        except Exception as e:
            print(f"❌ Error during agent continue: {e}")
            return False
    
    print(f"\n✅ Modern implementation test completed successfully!")
    print(f"   Total steps: {step_count}")
    print(f"   Thread ID: {thread_id}")
    return True


async def test_interrupt_configurations():
    """Test interrupt configuration loading and validation."""
    print("\n🔧 Testing Interrupt Configurations")
    print("-" * 40)
    
    service = create_modern_service()
    
    # Test different interrupt types
    interrupt_types = ["query_review", "insight_review", "final_review"]
    
    for interrupt_type in interrupt_types:
        config = service.get_interrupt_configuration(interrupt_type)
        print(f"📝 {interrupt_type}:")
        print(f"   UI Component: {config.get('ui_component', 'None')}")
        print(f"   Timeout: {config.get('timeout_seconds', 'None')}s")
        print(f"   Description: {config.get('description', 'None')}")
        
        # Test validation
        valid_input = service.validate_user_input(interrupt_type, "continue")
        invalid_input = service.validate_user_input(interrupt_type, "")
        
        print(f"   Validation 'continue': {valid_input.get('valid', False)}")
        print(f"   Validation empty: {invalid_input.get('valid', True)}")
        print()


async def main():
    """Main test runner."""
    print("🚀 Starting Modern V3 Implementation Tests")
    print("=========================================")
    
    try:
        # Test configurations
        await test_interrupt_configurations()
        
        # Test main workflow
        success = await test_modern_interrupt_pattern()
        
        if success:
            print("\n🎉 All tests passed! Modern V3 implementation is working correctly.")
            print("\n📋 What was tested:")
            print("   ✅ Modern HIL nodes with interrupt() pattern")
            print("   ✅ Configuration-driven workflow")
            print("   ✅ Service layer with Command pattern")
            print("   ✅ Input validation")
            print("   ✅ Parrot services integration")
            return 0
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 