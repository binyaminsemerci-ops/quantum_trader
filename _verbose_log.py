path = "/home/qt/quantum_trader/microservices/harvest_brain/harvest_brain.py"
content = open(path).read()
old = '                except Exception as e:\n                    logger.debug(f"Error evaluating position {pos_key}: {e}")'
new = '                except Exception as e:\n                    logger.error(f"TRACEBACK evaluating {pos_key}: {e}", exc_info=True)'
if old in content:
    content = content.replace(old, new, 1)
    open(path, "w").write(content)
    print("OK: verbose logging enabled")
else:
    print("NOT FOUND - searching for context...")
    idx = content.find("Error evaluating position")
    if idx > 0:
        print(repr(content[idx-50:idx+100]))
