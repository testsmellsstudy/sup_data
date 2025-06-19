   
import subprocess
import os
import numpy as np


repository = 'AISmessages'


class RunJacoco:

    def __init__(self):

        self._jacoco = ("~/.jdks/azul-22.0.2/bin/java -Dmaven.multiModuleProjectDirectory=$(pwd)/aismessages "
                        "-Djansi.passthrough=true -Dmaven.home=$HOME/.m2/wrapper/dists/apache-maven-3.5.0-bin/6ps54u5pnnbbpr6ds9rppcc7iv/apache-maven-3.5.0 "
                        "-Dclassworlds.conf=$HOME/.m2/wrapper/dists/apache-maven-3.5.0-bin/6ps54u5pnnbbpr6ds9rppcc7iv/apache-maven-3.5.0/bin/m2.conf "
                        "-Dmaven.ext.class.path=/snap/intellij-idea-community/553/plugins/maven/lib/maven-event-listener.jar "
                        "-javaagent:/snap/intellij-idea-community/553/lib/idea_rt.jar=37885:/snap/intellij-idea-community/553/bin "
                        "-Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 -Dsun.stderr.encoding=UTF-8 -classpath "
                        "$HOME/.m2/wrapper/dists/apache-maven-3.5.0-bin/6ps54u5pnnbbpr6ds9rppcc7iv/apache-maven-3.5.0/boot/plexus-classworlds-2.5.2.jar "
                        "org.codehaus.classworlds.Launcher -Didea.version=2024.3 clean test org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent "
                        "clean test org.jacoco:jacoco-maven-plugin:0.8.12:report")
        self.path = os.getcwd()

    def run_jacoco(self):
        print('Run JaCoCo ...')
        repositories = [repo for repo in os.listdir(self.path) if repo not in ['venv', 'run_jacoco.py']]

        # FAZ A VARREDURA NA PASTA DE DATASET #
        for repository in repositories:
            dataset = os.path.join(self.path, repository)
            print(dataset)

            # Defina um diretório de saída único para cada execução
            output_dir = os.path.join(dataset, "jacoco-report")
            os.makedirs(output_dir, exist_ok=True)

            # Run JaCoCo #
            try:
                os.chdir(dataset)
                print(f"\nRunning tests in: {repository}\n")
                env = os.environ.copy()
                env['JACOCO_OUTPUT_DIR'] = output_dir

                command = f"{self._jacoco} -Djacoco.outputDir={output_dir}"
                subprocess.check_call(command, shell=True, env=env)
                print(f"Report generated at {output_dir}\n")
            except subprocess.CalledProcessError as e:
                output = e.output
                print(f"Error running tests in {repository}:\n{output}\n")


if __name__ == '__main__':
    jacoco = RunJacoco()
    jacoco.run_jacoco()

