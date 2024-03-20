import subprocess

size_of_subset_list = [200, 500, 750, 1000, 1250]

for size_of_subset in size_of_subset_list:
    command = [
        'python', "script.py",
        '--size_of_subset', str(size_of_subset)
        ]

    subprocess.run(command)