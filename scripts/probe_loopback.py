# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
print("WASAPI:", wasapi)

print("\n=== default wasapi device (output) ===")
try:
    print(p.get_default_wasapi_device(output=True))
except Exception as e:
    print("err", type(e).__name__, e)

print("\n=== default wasapi device (input) ===")
try:
    print(p.get_default_wasapi_device(output=False))
except Exception as e:
    print("err", type(e).__name__, e)

print("\n=== get_default_wasapi_loopback ===")
try:
    print(p.get_default_wasapi_loopback())
except Exception as e:
    print("err", type(e).__name__, e)

print("\n=== loopback generator ===")
try:
    for info in p.get_loopback_device_info_generator():
        print(info)
except Exception as e:
    print("err", type(e).__name__, e)

print("\n=== all WASAPI outputs + analogue loopback ===")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get("hostApi") != wasapi["index"]:
        continue
    print(f"\nWASAPI dev {i}: {info}")
    if int(info.get("maxOutputChannels") or 0) > 0 and not info.get("isLoopbackDevice"):
        try:
            lb = p.get_wasapi_loopback_analogue_by_dict(info)
            print("  analogue loopback:", lb)
        except Exception as e:
            print("  analogue err:", type(e).__name__, e)
        try:
            lb2 = p.get_wasapi_loopback_analogue_by_index(i)
            print("  analogue by index:", lb2)
        except Exception as e:
            print("  analogue by index err:", type(e).__name__, e)

print("\n=== ALL devices isLoopback ===")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get("isLoopbackDevice") or (info.get("hostApi") == wasapi["index"]):
        print(i, "lb=", info.get("isLoopbackDevice"), "in=", info.get("maxInputChannels"),
              "out=", info.get("maxOutputChannels"), repr(info.get("name")))

p.terminate()

print("\n=== soundcard ===")
try:
    import soundcard as sc
    sp = sc.default_speaker()
    print("default speaker:", sp.name, sp.id)
    for s in sc.all_speakers():
        print(" speaker:", s.name)
    for mic in sc.all_microphones(include_loopback=True):
        print(" mic lb=", getattr(mic, "isloopback", None), mic.name)
    try:
        lb = sc.get_microphone(id=str(sp.name), include_loopback=True)
        print(" loopback for default:", lb.name, getattr(lb, "isloopback", None))
    except Exception as e:
        print(" get_microphone loopback err:", e)
except Exception as e:
    print("soundcard err", e)
