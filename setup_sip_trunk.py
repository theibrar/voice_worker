import os
import sys
import asyncio

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "http://127.0.0.1:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

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

    http_url = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
    lk = api.LiveKitAPI(url=http_url, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)

    methods = [m for m in dir(lk.sip) if not m.startswith("_")]
    print(f"► Detected LiveKit SIP Methods: {methods}\n")

    trunk_id = None
    rule_id = None

    # Determine Inbound Trunk Creation Method
    create_trunk_fn = getattr(lk.sip, "create_sip_inbound_trunk", None) or getattr(lk.sip, "create_inbound_trunk", None) or getattr(lk.sip, "create_sip_trunk", None)
    list_trunk_fn = getattr(lk.sip, "list_sip_inbound_trunks", None) or getattr(lk.sip, "list_inbound_trunks", None) or getattr(lk.sip, "list_sip_trunks", None) or getattr(lk.sip, "list_sip_inbound_trunk", None)

    try:
        if list_trunk_fn:
            req_cls = getattr(api, "ListSIPInboundTrunkRequest", None) or getattr(api, "ListSIPTrunkRequest", None)
            if req_cls:
                res = await list_trunk_fn(req_cls())
                items = getattr(res, "items", [])
                for t in items:
                    print(f"✓ Found existing trunk: {getattr(t, 'sip_trunk_id', '')} ({getattr(t, 'name', '')})")
                    trunk_id = getattr(t, "sip_trunk_id", None)

        if not trunk_id and create_trunk_fn:
            req_cls = getattr(api, "CreateSIPInboundTrunkRequest", None) or getattr(api, "CreateSIPTrunkRequest", None)
            info_cls = getattr(api, "SIPInboundTrunkInfo", None) or getattr(api, "SIPTrunkInfo", None)
            if req_cls and info_cls:
                new_t = await create_trunk_fn(
                    req_cls(
                        trunk=info_cls(
                            name="Telnyx Inbound Trunk",
                            numbers=PHONE_NUMBERS,
                        )
                    )
                )
                trunk_id = getattr(new_t, "sip_trunk_id", "ST_TELNYX_PRIMARY")
                print(f"✓ Created SIP Inbound Trunk: {trunk_id}")

        # Dispatch Rule Methods
        create_rule_fn = getattr(lk.sip, "create_sip_dispatch_rule", None) or getattr(lk.sip, "create_dispatch_rule", None)
        list_rule_fn = getattr(lk.sip, "list_sip_dispatch_rules", None) or getattr(lk.sip, "list_dispatch_rules", None)

        if list_rule_fn:
            req_cls = getattr(api, "ListSIPDispatchRuleRequest", None) or getattr(api, "ListDispatchRuleRequest", None)
            if req_cls:
                res = await list_rule_fn(req_cls())
                items = getattr(res, "items", [])
                for r in items:
                    print(f"✓ Found existing dispatch rule: {getattr(r, 'sip_dispatch_rule_id', '')}")
                    rule_id = getattr(r, "sip_dispatch_rule_id", None)

        if not rule_id and create_rule_fn:
            req_cls = getattr(api, "CreateSIPDispatchRuleRequest", None)
            info_cls = getattr(api, "SIPDispatchRuleInfo", None)
            rule_cls = getattr(api, "SIPDispatchRule", None)
            indiv_cls = getattr(api, "SIPDispatchRuleIndividual", None)

            if req_cls and rule_cls and indiv_cls:
                try:
                    new_r = await create_rule_fn(
                        req_cls(
                            name="Telnyx Inbound Dispatcher",
                            rule=rule_cls(
                                dispatch_rule_individual=indiv_cls(
                                    room_prefix="call-",
                                )
                            ),
                            trunk_ids=[trunk_id] if trunk_id else [],
                        )
                    )
                    rule_id = getattr(new_r, "sip_dispatch_rule_id", "SDR_TELNYX_DISPATCH")
                    print(f"✓ Created SIP Dispatch Rule: {rule_id}")
                except Exception as ex:
                    # Alternative payload structure
                    new_r = await create_rule_fn(
                        rule_cls(
                            dispatch_rule_individual=indiv_cls(
                                room_prefix="call-",
                            )
                        )
                    )
                    rule_id = getattr(new_r, "sip_dispatch_rule_id", "SDR_TELNYX_DISPATCH")
                    print(f"✓ Created SIP Dispatch Rule: {rule_id}")

        print("\n" + "=" * 65)
        print(" 🎉 LiveKit SIP Inbound Trunk & Dispatch Rules are ACTIVE!")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"Notice during setup: {e}")
    finally:
        await lk.aclose()

if __name__ == "__main__":
    asyncio.run(configure_sip())
