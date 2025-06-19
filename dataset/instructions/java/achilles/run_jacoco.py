import subprocess
import os
import numpy as np


class RunJacoco:

    def __init__(self):
        self._jacoco = (
            '~/.jdks/corretto-1.8.0_432/bin/java -Dmaven.multiModuleProjectDirectory=$(pwd)/achilles -Dnonesi.passthrough=true -Dmaven.home=/snap/intellij-idea-community/553/plugins/maven/lib/maven3 -Dclassworlds.conf=/snap/intellij-idea-community/553/plugins/maven/lib/maven3/bin/m2.conf -Dmaven.ext.class.path=/snap/intellij-idea-community/553/plugins/maven/lib/maven-event-listener.jar -javaagent:/snap/intellij-idea-community/553/lib/idea_rt.jar=36297:/snap/intellij-idea-community/553/bin -Dfile.encoding=UTF-8 -classpath /snap/intellij-idea-community/553/plugins/maven/lib/maven3/boot/plexus-classworlds-2.8.0.jar:/snap/intellij-idea-community/553/plugins/maven/lib/maven3/boot/plexus-classworlds.license org.codehaus.classworlds.Launcher -Didea.version=2024.3 clean test org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent clean test org.jacoco:jacoco-maven-plugin:0.8.12:report')

        self.path = os.getcwd()

    def run_jacoco(self):
        print('Run JaCoCo Wikidata...')
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

