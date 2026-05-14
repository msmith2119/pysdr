import sys
from command_loop import DSLContext   # ← change to your module name

def main():
    if len(sys.argv) != 2:
        print("Usage: python driver.py <script-file>")
        sys.exit(1)

    script_path = sys.argv[1]

    ctx = DSLContext()
    ctx.runFile(script_path)

if __name__ == "__main__":
    main()
