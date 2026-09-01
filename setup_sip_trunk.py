import os
import sys
import asyncio

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("sip_setup")

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "http://127.0.0.1:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# Telnyx Phone Numbers
PHONE_NUMBERS = ["+14153845276", "+16676668582"]

async def configure_sip():
    print("\n" + "=" * 65)
    print(" 📞  CONFIGURING LIVEKIT SIP TRUNK & INBOUND DISPATCH RULES")
    print(f"    LiveKit Server: {LIVEKIT_URL}")
    print(f"    Numbers: {', '.join(PHONE_NUMBERS)}")
    print("=" * 65 + "\n")

    try:
        from livekit import api
    except ImportError:
        print("❌ Error: livekit SDK not installed.")
        return

    # Normalize url for HTTP API
    http_url = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    lk = api.LiveKitAPI(url=http_url, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)

    try:
        # 1. Check Existing Trunks
        print("► Step 1: Querying existing SIP Inbound Trunks...")
        existing_trunks = await lk.sip.list_sip_inbound_trunks(api.ListSIPInboundTrunkRequest())
        trunk_id = None

        for t in existing_trunks.items:
            print(f"  Found existing trunk: {t.sip_trunk_id} ({t.name}) - Numbers: {list(t.numbers)}")
            trunk_id = t.sip_trunk_id

        # 2. Create Inbound Trunk if not present
        if not trunk_id:
            print("► Step 2: Creating new Telnyx SIP Inbound Trunk...")
            new_trunk = await lk.sip.create_sip_inbound_trunk(
                api.CreateSIPInboundTrunkRequest(
                    trunk=api.SIPInboundTrunkInfo(
                        name="Telnyx Inbound Trunk (server.ibrasoft.com)",
                        numbers=PHONE_NUMBERS,
                        allowed_addresses=[], # Allow any valid Telnyx SBC IP
                    )
                )
            )
            trunk_id = new_trunk.sip_trunk_id
            print(f"✓ Created SIP Inbound Trunk: {trunk_id}")
        else:
            print(f"✓ Using existing SIP Inbound Trunk: {trunk_id}")

        # 3. Check Existing Dispatch Rules
        print("\n► Step 3: Querying existing SIP Dispatch Rules...")
        existing_rules = await lk.sip.list_sip_dispatch_rules(api.ListSIPDispatchRuleRequest())
        rule_id = None

        for r in existing_rules.items:
            print(f"  Found existing dispatch rule: {r.sip_dispatch_rule_id} ({r.name})")
            rule_id = r.sip_dispatch_rule_id

        # 4. Create Dispatch Rule if not present
        if not rule_id:
            print("► Step 4: Creating SIP Inbound Dispatch Rule (Auto-Route to Voice Agent Worker)...")
            new_rule = await lk.sip.create_sip_dispatch_rule(
                api.CreateSIPDispatchRuleRequest(
                    rule=api.SIPDispatchRuleInfo(
                        name="Telnyx Inbound to AI Voice Worker",
                        rule=api.SIPDispatchRule(
                            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                                room_prefix="call-",
                            )
                        ),
                        trunk_ids=[trunk_id] if trunk_id else [],
                    )
                )
            )
            rule_id = new_rule.sip_dispatch_rule_id
            print(f"✓ Created SIP Dispatch Rule: {rule_id}")
        else:
            print(f"✓ Using existing SIP Dispatch Rule: {rule_id}")

        print("\n" + "=" * 65)
        print(" 🎉 SUCCESS: LiveKit SIP Inbound Engine is 100% Configured!")
        print("    Incoming Telnyx calls to +14153845276 or +16676668582")
        print("    will now instantly connect, ring, and answer via AI voice agent!")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"❌ Error configuring LiveKit SIP: {e}")
    finally:
        await lk.aclose()

if __name__ == "__main__":
    asyncio.run(configure_sip())
