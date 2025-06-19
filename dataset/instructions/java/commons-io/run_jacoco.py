import subprocess
import re
import os
import numpy as np

class runJacoco:

    def __init__(self):


        self._jacoco = ('~/.jdks/corretto-22.0.2/bin/java -Dmaven.multiModuleProjectDirectory=$(pwd)/commons-io '
                        '-Djansi.passthrough=true -Dmaven.home=/snap/intellij-idea-community/553/plugins/maven/lib/maven3 '
                        '-Dclassworlds.conf=/snap/intellij-idea-community/553/plugins/maven/lib/maven3/bin/m2.conf '
                        '-Dmaven.ext.class.path=/snap/intellij-idea-community/553/plugins/maven/lib/maven-event-listener.jar '
                        '-javaagent:/snap/intellij-idea-community/553/lib/idea_rt.jar=38141:/snap/intellij-idea-community/553/bin '
                        '-Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 -Dsun.stderr.encoding=UTF-8 -classpath '
                        '/snap/intellij-idea-community/553/plugins/maven/lib/maven3/boot/plexus-classworlds-2.8.0.jar:/snap/intellij-idea-community/553/plugins/maven/lib/maven3/boot/plexus-classworlds.license '
                        'org.codehaus.classworlds.Launcher -Didea.version=2024.3 '
                        'clean test org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent '
                        'clean test org.jacoco:jacoco-maven-plugin:0.8.12:report')
        self.path = os.getcwd()

    def run_jacoco(self):
        print('Run JaCoCo Commons-io... ')
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

