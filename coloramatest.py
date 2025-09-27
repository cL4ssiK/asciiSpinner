import colorama
import time, sys

colorama.init()


n = 20
print("Line " + "\nLine ".join(str(i) for i in range(n)))
time.sleep(1)
# Move back up n lines
sys.stdout.write(f"\033[{n}A")
sys.stdout.write(">>> Overwrote top line!\n")
sys.stdout.flush()
