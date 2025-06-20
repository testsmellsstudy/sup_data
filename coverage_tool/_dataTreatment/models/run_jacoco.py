import subprocess
import re
import os
import numpy as np

class runJacoco:

    def __init__(self):

        self._jacoco = ('mvn org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent test '
                        'org.jacoco:jacoco-maven-plugin:0.8.12:report')
        self.path = os.getcwd()

    def run_jacoco(self):
        print('Run JaCoCo...  ')
        repositories = np.asarray(os.listdir(self.path)).tolist()

        # FAZ A VARREDURA NA PASTA DE DATASET #
        for repository in repositories:
            if repository == 'venv' or repository == 'run_jacoco.py':
                continue
            dataset = os.path.join(self.path, repository)
            print(dataset)

            # Run JaCoCo #
            try:
                os.chdir(dataset)
                print(f"\nRunning tests in: {repository}\n")
                subprocess.check_output(self._jacoco, shell=True, encoding="utf-8", text=True)
            except subprocess.CalledProcessError as e:
                output = e.output
                print(output)


if __name__ == '__main__':

    jacoco = runJacoco()
    jacoco.run_jacoco()
