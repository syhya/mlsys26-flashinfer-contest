import subprocess

def run_tests():
    result = subprocess.run(["python3", "task_info.py"], capture_output=True, text=True)
    print("Task info:")
    print(result.stdout)
    
if __name__ == "__main__":
    run_tests()
