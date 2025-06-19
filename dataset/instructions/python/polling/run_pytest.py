import subprocess
import numpy as np
import os

project = 'polling'


class runPytest:

    def __init__(self):

        self._pytest = (f'pytest --cov --cov-report html --cov-report term-missing --cov-report json '
                        f'--cov-branch --report-log {project}.log')
        self.path = os.getcwd()

    def run_pytest(self):
        print(f'\nRun pytest in {project}...\n')
        repositories = np.asarray(os.listdir(self.path)).tolist()

        # FAZ A VARREDURA NA PASTA DE DATASET #
        for repository in repositories:
            if repository == 'venv' or repository == 'run_pytest.py':
                continue
            dataset = os.path.join(self.path, repository)
            print(dataset)

            # EXECUTA O PYTEST #
            try:
                os.chdir(dataset)
                print("\n")
                print(150 * "*")
                print(f"\nRunning Tests in: {repository}\n")
                print(150 * "*")
                print("\n")
                subprocess.check_output(self._pytest, shell=True, encoding="utf-8")
            except subprocess.CalledProcessError as e:
                output = e.output
                print(output)


if __name__ == '__main__':
    pytest = runPytest()
    pytest.run_pytest()
