import subprocess


size_of_subset = 1000
command = [
    'python', "script.py",
    '--size_of_subset', str(size_of_subset)
    ]

subprocess.run(command)