import subprocess

size_of_subset_list = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250]

for size_of_subset in size_of_subset_list:
    print("Running on subset of size", size_of_subset)
    command = [
        'python', "script.py",
        '--size_of_subset', str(size_of_subset)
        ]

    subprocess.run(command)