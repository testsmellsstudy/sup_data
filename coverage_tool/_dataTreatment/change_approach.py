import os
import shutil
from importlib.metadata import version
import pandas as pd
import logging
from _dataTreatment.models.curingas import list_folders_to_menu
from python.coverage._instructions.stig.run_pytest import project

ROOT = os.getcwd()
parent_dir = os.path.dirname(ROOT)

# Logger Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_directory(directory_path):
    """Create a directory if it doesn't exist'."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.debug(f"Directory created or already existing: {directory_path}")
    except OSError as e:
        logging.error(f"Error creating directory: {directory_path}. Erro: {e}")
        raise


def copy_file_to_target(source_file, target_dir):
    """Copy the file to the destination directory."""
    try:
        shutil.copy(source_file, target_dir)
        logging.info(f"File copied from {source_file} para {target_dir}")
    except shutil.Error as e:
        logging.error(f"Error to coping file: {source_file} for {target_dir}. Error: {e}")
        raise


def organize_csv(base_path):
    """
    Organizes CSV files from directories into subdirectories based on filenames.

    :param base_path: Path of the base directory containing the subdirectories.
    """
    try:
        # Filters subdirectories ignoring those starting with '0'
        subdirectories = [d for d in os.listdir(base_path) if
                          os.path.isdir(os.path.join(base_path, d)) and not d.startswith('0')]

        # Main directory "bySmell"
        by_smell_dir = os.path.join(base_path, "bySmell")
        create_directory(by_smell_dir)

        for subdir in subdirectories:
            subdir_path = os.path.join(base_path, subdir)

            # Iterates through the files within the subdirectory
            for file_name in os.listdir(subdir_path):
                # Check if it is a file
                source_file = os.path.join(subdir_path, file_name)
                if not os.path.isfile(source_file):
                    logging.debug(f"Ignoring {source_file}, because it is not a file.")
                    continue

                # Extract the name of the "smell" from the file
                smell_name = file_name.split('.')[-2]

                # Create directory for "smell" if needed
                smell_dir = os.path.join(by_smell_dir, smell_name)
                create_directory(smell_dir)

                #Copy the file to the "smell" directory
                copy_file_to_target(source_file, smell_dir)

        logging.info("CSV file organization completed successfully.")

    except Exception as e:
        logging.error(f"Error organizing CSV files. Error: {e}")
        raise


def aggregate_values_by_smell(input_directory, output_directory, language):

    """
    Traverses a directory, processes .csv files in subdirectories, and saves results to a .csv file.

    Args:
        input_directory (str): Input directory where subdirectories and .csv files are located.
        output_directory (str): Directory where the consolidated file will be saved.
    """

    os.makedirs(output_directory, exist_ok=True)

    if not os.path.exists(input_directory):
        print(f"Path to directory '{input_directory}'\n\n")
        print(f"Error: The directory '{input_directory.split('/')[-1]}' does not exist. Please check the path and try again.\n")

    for root, dirs, files in os.walk(input_directory):
        results = []
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                # Read .csv file
                try:
                    if language == 'python':
                        df = pd.read_csv(file_path)
                        # Performs the calculations
                        sum_statements  = df["statements_diff"].sum() if "statements_diff" in df.columns else 0
                        sum_missing     = df["missing_diff"].sum() if "missing_diff" in df.columns else 0
                        mean_statements = df["statements_diff"].mean() if "statements_diff" in df.columns else 0
                        mean_missing    = df["missing_diff"].mean() if "missing_diff" in df.columns else 0

                        # Store the result
                        results.append({
                            "file_name":        file,
                            "sum_statements":   int(sum_statements),
                            "sum_missing":      int(sum_missing),
                            "mean_statements":  float(mean_statements),
                            "mean_missing":     float(mean_missing)
                        })
                        smell_name = file.split('.')[-2]
                        if not os.path.exists(os.path.join(output_directory, smell_name)):
                            os.makedirs(os.path.join(output_directory, smell_name), exist_ok=True)

                        consolidated_df = pd.DataFrame(results)
                        output_file_path = os.path.join(output_directory, smell_name,
                                                        f"{smell_name}_consolidated_results.csv")
                        consolidated_df.to_csv(output_file_path, index=False)

                    elif language == 'java':
                        df = pd.read_csv(file_path)
                        # Performs the calculations
                        sum_statements  = df["LINE_COVERED"].sum() if "LINE_COVERED" in df.columns else 0
                        sum_missing     = df["LINE_MISSED"].sum() if "LINE_MISSED" in df.columns else 0
                        mean_statements = df["LINE_COVERED"].mean() if "LINE_COVERED" in df.columns else 0
                        mean_missing    = df["LINE_MISSED"].mean() if "LINE_MISSED" in df.columns else 0

                        # Store the result
                        results.append({
                            "file_name": file,
                            "sum_statements":   int(sum_statements),
                            "sum_missing":      int(sum_missing),
                            "mean_statements":  float(mean_statements),
                            "mean_missing":     float(mean_missing)
                        })
                        smell_name = file.split('.')[-2]
                        if not os.path.exists(os.path.join(output_directory, smell_name)):
                            os.makedirs(os.path.join(output_directory, smell_name), exist_ok=True)

                        consolidated_df = pd.DataFrame(results)
                        output_file_path = os.path.join(output_directory, smell_name,
                                                        f"{smell_name}_consolidated_results.csv")
                        consolidated_df.to_csv(output_file_path, index=False)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")


def process_csv_files(input_dir, output_file):
    """
    It traverses the directory and its subdirectories, processes .csv files, and creates a new consolidated .csv file.

    Args:
        input_dir (str): Directory path to be traversed.
        output_file (str): Path of the consolidated file to be generated.
    """
    consolidated_data = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]

                try:

                    df = pd.read_csv(file_path)

                    sum_statements  = df['sum_statements'].sum() if 'sum_statements' in df.columns else 0
                    sum_missing     = df['sum_missing'].sum() if 'sum_missing' in df.columns else 0

                    consolidated_data.append({
                        'file_name':        file_name,
                        'sum_statements':   sum_statements,
                        'sum_missing':      sum_missing
                    })

                except Exception as e:
                    print(f"Erro ao processar o arquivo {file_path}: {e}")

    consolidated_df = pd.DataFrame(consolidated_data)

    consolidated_df.to_csv(output_file, index=False)
    print(f"Arquivo consolidado salvo em: {output_file}")


def main_menu():
    """
    Displays the main menu, navigates through directory options, and organizes CSV files
    based on the user's selections.
    """
    try:
        language = list_folders_to_menu(parent_dir, word='Language')
        option = int(input(
            'Do you want to generate? '
            '\n [1] - Extract Results'
            '\n [2] - Difference'
            '\n [3] - Aggregate Values by Smell \n\n'
        ))

        language_name = language.split('/')[-1]
    except ValueError:
        print("Error: Invalid input. Please enter a number corresponding to the options.")
        return

    def navigate_folders(base_path, *steps):
        """Helper function to navigate through folders based on the provided steps."""
        path = base_path
        for step in steps:
            path = list_folders_to_menu(path, word=step)
        return path

    def process_option(base_path, steps, process_func, *args):
        """Helper function to process options with reusable logic."""
        version = navigate_folders(base_path, *steps)
        process_func(version, *args)

    if option == 1:
        print("Option 1 - Extract Results\n")
        base_path = os.path.join(language, 'coverage', 'extraction_results')
        process_option(base_path, ['Dataset', 'Model', 'Version'], organize_csv)

    elif option == 2:
        print("Option 2 - Difference\n")
        base_path = os.path.join(language, 'coverage', 'data_analysis/_difference')
        process_option(base_path, ['Dataset', 'Model', 'Version'], organize_csv)

    elif option == 3:
        print("Option 3 - Aggregate Values by Smell\n")
        base_path   = os.path.join(language, 'coverage', 'data_analysis/_difference')
        version     = navigate_folders(base_path, 'Dataset', 'Model', 'Version')
        input_dir   = os.path.join(version, 'bySmell')

        # Extract metadata for output paths
        version_name    = version.split('/')[-1]
        model_name      = version.split('/')[-2]
        approach        = version.split('/')[-3]
        output_dir      = os.path.join(base_path, f"_total/refactored/{approach}/bySmell/{version_name}/{model_name}")
        aggregate_values_by_smell(input_dir, output_dir, language_name)

        # Process consolidated CSV
        input_directory = os.path.join(
            parent_dir, language_name, 'coverage', 'data_analysis/_difference/_total/refactored',
            approach, 'bySmell', version_name, model_name
        )
        output_csv = os.path.join(
            parent_dir, language_name, 'coverage', 'results', model_name,
            f"{language_name}-{model_name}_consolidated-results-bySmell.csv"
        )
        process_csv_files(input_directory, output_csv)
    else:
        print("Option not recognized. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()








































# import os
# import shutil
# from importlib.metadata import version
# import pandas as pd
# import logging
# from _dataTreatment.models.curingas import list_folders_to_menu
# from python.coverage._instructions.stig.run_pytest import project
#
# ROOT = os.getcwd()
# parent_dir = os.path.dirname(ROOT)
#
# # Configuração do logger
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#
#
# def create_directory(directory_path):
#     """Cria um diretório se ele não existir."""
#     try:
#         os.makedirs(directory_path, exist_ok=True)
#         logging.debug(f"Diretório criado ou já existente: {directory_path}")
#     except OSError as e:
#         logging.error(f"Erro ao criar diretório: {directory_path}. Erro: {e}")
#         raise
#
#
# def copy_file_to_target(source_file, target_dir):
#     """Copia o arquivo para o diretório de destino."""
#     try:
#         shutil.copy(source_file, target_dir)
#         logging.info(f"Arquivo copiado de {source_file} para {target_dir}")
#     except shutil.Error as e:
#         logging.error(f"Erro ao copiar o arquivo: {source_file} para {target_dir}. Erro: {e}")
#         raise
#
#
# def organize_csv(base_path):
#     """
#     Organiza arquivos CSV de diretórios em subdiretórios baseados no nome dos arquivos.
#
#     :param base_path: Caminho do diretório base que contém os subdiretórios.
#     """
#     try:
#         # Filtra os subdiretórios ignorando os que começam com '0'
#         subdirectories = [d for d in os.listdir(base_path) if
#                           os.path.isdir(os.path.join(base_path, d)) and not d.startswith('0')]
#
#         # Diretório principal "bySmell"
#         by_smell_dir = os.path.join(base_path, "bySmell")
#         create_directory(by_smell_dir)
#
#         for subdir in subdirectories:
#             subdir_path = os.path.join(base_path, subdir)
#
#             # Itera pelos arquivos dentro do subdiretório
#             for file_name in os.listdir(subdir_path):
#                 # Verifica se é um arquivo
#                 source_file = os.path.join(subdir_path, file_name)
#                 if not os.path.isfile(source_file):
#                     logging.debug(f"Ignorando {source_file}, pois não é um arquivo.")
#                     continue
#
#                 # Extrai o nome do "smell" do arquivo
#                 smell_name = file_name.split('.')[-2]
#
#                 # Cria o diretório para o "smell" se necessário
#                 smell_dir = os.path.join(by_smell_dir, smell_name)
#                 create_directory(smell_dir)
#
#                 # Copia o arquivo para o diretório do "smell"
#                 copy_file_to_target(source_file, smell_dir)
#
#         logging.info("Organização dos arquivos CSV concluída com sucesso.")
#
#     except Exception as e:
#         logging.error(f"Erro ao organizar os arquivos CSV. Erro: {e}")
#         raise
#
#
# def aggragete_values_by_smell(input_directory, output_directory):
#
#     """
#     Percorre um diretório, processa arquivos .csv em subdiretórios e salva resultados em um arquivo .csv.
#
#     Args:
#         input_directory (str): Diretório de entrada onde estão os subdiretórios e arquivos .csv.
#         output_directory (str): Diretório onde o arquivo consolidado será salvo.
#     """
#
#     os.makedirs(output_directory, exist_ok=True)
#
#     # consolidated_df = pd.DataFrame(columns=["file_name", "sum_statement", "sum_missing", "mean_statements", "mean_missing"])
#     for root, dirs, files in os.walk(input_directory):
#         results = []
#         for file in files:
#             if file.endswith(".csv"):
#                 file_path = os.path.join(root, file)
#                 # Read .csv file
#                 try:
#                     df = pd.read_csv(file_path)
#                     # Performs the calculations
#                     sum_statements  = df["statements_diff"].sum() if "statements_diff" in df.columns else 0
#                     sum_missing     = df["missing_diff"].sum() if "missing_diff" in df.columns else 0
#                     mean_statements = df["statements_diff"].mean() if "statements_diff" in df.columns else 0
#                     mean_missing    = df["missing_diff"].mean() if "missing_diff" in df.columns else 0
#
#                     # Store the result
#                     results.append({
#                         "file_name":        file,
#                         "sum_statements":   int(sum_statements),
#                         "sum_missing":      int(sum_missing),
#                         "mean_statements":  float(mean_statements),
#                         "mean_missing":     float(mean_missing)
#                     })
#                     smell_name = file.split('.')[-2]
#                     if not os.path.exists(os.path.join(output_directory, smell_name)):
#                         os.makedirs(os.path.join(output_directory, smell_name), exist_ok=True)
#
#                     consolidated_df = pd.DataFrame(results)
#                     output_file_path = os.path.join(output_directory, smell_name,
#                                                     f"{smell_name}_consolidated_results.csv")
#                     consolidated_df.to_csv(output_file_path, index=False)
#
#                 except Exception as e:
#                     print(f"Error processing file {file_path}: {e}")
#
#
# def process_csv_files(input_dir, output_file):
#     """
#     Percorre o diretório e seus subdiretórios, processa arquivos .csv e cria um novo arquivo .csv consolidado.
#
#     Args:
#         input_dir (str): Caminho do diretório a ser percorrido.
#         output_file (str): Caminho do arquivo consolidado a ser gerado.
#     """
#     # Lista para armazenar os dados consolidados
#     consolidated_data = []
#
#     # Percorre o diretório e subdiretórios
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             # Verifica se é um arquivo .csv
#             if file.endswith(".csv"):
#                 file_path = os.path.join(root, file)
#                 file_name = os.path.splitext(file)[0]  # Remove a extensão do nome do arquivo
#
#                 try:
#                     # Lê o arquivo CSV
#                     df = pd.read_csv(file_path)
#
#                     # Calcula as somas das colunas necessárias
#                     sum_statements = df['sum_statements'].sum() if 'sum_statements' in df.columns else 0
#                     sum_missing = df['sum_missing'].sum() if 'sum_missing' in df.columns else 0
#
#                     # Adiciona os dados ao consolidado
#                     consolidated_data.append({
#                         'file_name': file_name,
#                         'sum_statements': sum_statements,
#                         'sum_missing': sum_missing
#                     })
#
#                 except Exception as e:
#                     print(f"Erro ao processar o arquivo {file_path}: {e}")
#
#     # Cria um DataFrame consolidado
#     consolidated_df = pd.DataFrame(consolidated_data)
#
#     # Salva o DataFrame consolidado em um arquivo .csv
#     consolidated_df.to_csv(output_file, index=False)
#     print(f"Arquivo consolidado salvo em: {output_file}")
#
#
# def main_menu():
#     """
#     Displays the main menu, navigates through directory options, and organizes CSV files
#     based on the user's selections.
#     """
#     try:
#
#         language = list_folders_to_menu(parent_dir, word='Language')
#         option = int(input('Do you want to generate? '
#                            '\n [1] - Extract Results'
#                            '\n [2] - Difference'
#                            '\n [3] - Aggregate Values by Smell \n\n'))
#
#     except ValueError as e:
#         print(f"Error: {str(e)}")
#         return
#
#     if option == 1:
#         print("Option 1 - Extract Results\n")
#
#         extraction_results_path = os.path.join(language, 'coverage', 'extraction_results')
#         dataset = list_folders_to_menu(extraction_results_path, word='Dataset')
#         model = list_folders_to_menu(dataset, word='Model')
#         version = list_folders_to_menu(model, word='Version')
#         organize_csv(version)
#
#     elif option == 2:
#         print("Option 2 - Difference\n")
#         extraction_results_path_diff = os.path.join(language, 'coverage', 'data_analysis/_difference')
#         dataset = list_folders_to_menu(extraction_results_path_diff, word='Dataset')
#         model = list_folders_to_menu(dataset, word='Model')
#         version = list_folders_to_menu(model, word='Version')
#         organize_csv(version)
#
#     elif option == 3:
#         print("Option 3 - Aggregate Values by Smell\n")
#         extraction_results_path_diff = os.path.join(language, 'coverage', 'data_analysis/_difference')
#         dataset = list_folders_to_menu(extraction_results_path_diff, word='Dataset')
#         model = list_folders_to_menu(dataset, word='Model')
#         version = list_folders_to_menu(model, word='Version')
#         input_dir = f"{version}/bySmell"
#         version_name = version.split('/')[-1]
#         model_name = model.split('/')[-1]
#         approach = dataset.split('/')[-1]
#         print(approach)
#         output_dir = os.path.join((extraction_results_path_diff), f"_total/refactored/{approach}/bySmell/{version_name}/{model_name}")
#         aggragete_values_by_smell(input_dir, output_dir)
#
#     else:
#         print("Option not recognized. Please select 1, 2, or 3.")
#
#
# if __name__ == "__main__":
#
#     # main_menu()
#     LLM = 'GEMINI'
#     input_directory = f"{parent_dir}/python/coverage/data_analysis/_difference/_total/refactored/Final/bySmell/VF/{LLM}"  # Substitua pelo caminho do diretório que deseja percorrer
#     output_csv = f"{LLM}_consolidated_results.csv"  # Nome do arquivo consolidado
#     process_csv_files(input_directory, output_csv)
#
#
#
