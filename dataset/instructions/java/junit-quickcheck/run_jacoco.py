import subprocess
import re
import os
import numpy as np

# OSB.: Não precisa adicionar o plugin no arquivo pom.xml

class runJacoco:

    def __init__(self):


        self._jacoco = ('~/.jdks/corretto-1.8.0_432/bin/java -Dmaven.multiModuleProjectDirectory=$(pwd)/junit-quickcheck '
                        '-Djansi.passthrough=true -Dmaven.home=/snap/intellij-idea-community/553/plugins/maven/lib/maven3 '
                        '-Dclassworlds.conf=/snap/intellij-idea-community/553/plugins/maven/lib/maven3/bin/m2.conf '
                        '-Dmaven.ext.class.path=/snap/intellij-idea-community/553/plugins/maven/lib/maven-event-listener.jar '
                        '-javaagent:/snap/intellij-idea-community/553/lib/idea_rt.jar=42279:/snap/intellij-idea-community/553/bin '
                        '-Dfile.encoding=UTF-8 -classpath /snap/intellij-idea-community/553/plugins/maven/lib/maven3/boot/plexus-classworlds-2.8.0.jar:/snap/intellij-idea-community/553/plugins/maven/lib/maven3/boot/plexus-classworlds.license '
                        'org.codehaus.classworlds.Launcher -Didea.version=2024.3 '
                        'clean test org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent '
                        'clean test org.jacoco:jacoco-maven-plugin:0.8.12:report')

        self.path = os.getcwd()

    def run_jacoco(self):
        print('Run JaCoCo JUnit-QuickCheck...  ')
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

