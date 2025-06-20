import os
import pathlib
import shutil
from collections import defaultdict
from typing import Tuple, List, Any
import pandas as pd

from _dataTreatment.models.curingas import organize_directories, list_folders_to_menu, get_split_name

ROOT = os.getcwd()
parent_dir = os.path.dirname(ROOT)


def copy_html_files(root_dir, destination_dir):
    """
    function that copies html files from one folder to another
    :param root_dir:
    :param destination_dir:
    :return:
    """

    html_ref = "htmlcov"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    num_files = 0
    for root, dirs, files in os.walk(root_dir):
        if html_ref in dirs:
            html_cov_path = os.path.join(root, html_ref)
            index_file = os.path.join(html_cov_path, 'index.html')
            if os.path.isfile(index_file):
                num_files += 1
                # Determine the name of the main folder
                main_folder_name = os.path.basename(root)
                main_folder_name = main_folder_name.split('_')[0]
                # Create the destination subfolder
                dest_subfolder = os.path.join(destination_dir, main_folder_name)
                if not os.path.exists(dest_subfolder):
                    os.makedirs(dest_subfolder)
                # Copy the index.html file to the destination subfolder
                shutil.copy(index_file, os.path.join(dest_subfolder, 'index.html'))
                print(f"Copied {index_file} to {os.path.join(dest_subfolder, 'index.html')}")
            # Avoid descending into 'htmlcov' directory
            dirs.remove(html_ref)
    print(f'\n{num_files} copied files!')


def dataset_management(language):
    approach_name = list_folders_to_menu(os.path.join(language, 'dataset/refactored'), word='Approach')
    llm_model = list_folders_to_menu(approach_name, word='LLM')
    version_number = list_folders_to_menu(llm_model, word='Version')

    organize_directories(version_number, language)
    print('Folders organized!!!')


def find_jacoco_files(root_directory) -> tuple[list[tuple[Any, str]], int]:
    jacoco_files = []
    site_count = 0

    for subdir, dirs, files in os.walk(root_directory):
        # print(subdir)
        # Verifica se o diretório atual é o "target"
        if os.path.basename(subdir) == "target":
            site_path = os.path.join(subdir, "site")
            if os.path.isdir(site_path):
                site_count += 1

            jacoco_path = os.path.join(site_path, "jacoco")
            jacoco_aggregate_path = os.path.join(site_path, "jacoco-aggregate")

            jacoco_csv_path = os.path.join(jacoco_path, "jacoco.csv")
            jacoco_aggregate_csv_path = os.path.join(jacoco_aggregate_path, "jacoco.csv")

            if os.path.isfile(jacoco_csv_path):
                jacoco_files.append((subdir, jacoco_csv_path))

            elif os.path.isfile(jacoco_aggregate_csv_path):
                jacoco_files.append((subdir, jacoco_aggregate_csv_path))

    return jacoco_files


def count_target_directories_and_subdirs(root_directory):
    target_counts = defaultdict(int)
    direct_subdirs = 0
    total_targets = 0

    is_root = True

    for subdir, dirs, files in os.walk(root_directory):
        if is_root:
            direct_subdirs = len(dirs)
            is_root = False

        # Conta quantos "target" existem em cada subdiretório
        target_count = dirs.count("target")
        if target_count > 0:
            parent_dir = os.path.abspath(subdir)
            target_counts[parent_dir] += target_count

    for parent_dir, count in target_counts.items():
        total_targets += count

    return total_targets, direct_subdirs


def aggregate(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        csv_file_path = pd.read_csv(os.path.join(path, file))
        df = pd.concat([df, csv_file_path])
    # if original:
    if not os.path.isfile(f'{path}/jacoco.csv'):
        df.to_csv(f'{path}/jacoco.csv', index=False)
    else:
        print('\n Jacoco file already exists!!! Please, delete the file and try again.\n')
    # else:
    #     print(origin_folder_name)
    #     if not os.path.isfile(f'{path}/{origin_folder_name}_jacoco.csv'):
    #         df.to_csv(f'{path}/{origin_folder_name}_jacoco.csv', index=False)
    #     else:
    #         print('\n Jacoco file already exists!!! Please, delete the file and try again.\n')


def copy_original_jacoco_files(root_directory, file_destination):
    jacoco_files = find_jacoco_files(root_directory)
    if jacoco_files:
        for target_path, jacoco_file in jacoco_files:
            origin_folder_name = os.path.basename(os.path.dirname(target_path))
            destin_folder = os.path.join(file_destination, root_directory.split('/')[-1])
            os.makedirs(destin_folder, exist_ok=True)
            file_name = os.path.join(destin_folder, f"{origin_folder_name}_jacoco.csv")
            shutil.copy2(jacoco_file, file_name)
        aggregate(destin_folder)
        print(f"Jacoco '.csv' files copied to: {file_destination}")

    else:
        print("File 'jacoco.csv' not found!!!")


def copy_jacoco_files(root_directory, file_destination):

    for refactored_project in os.listdir(root_directory):
        if os.path.isdir(os.path.join(root_directory, refactored_project)):
            directory = os.path.join(root_directory, refactored_project)
            target_counts, total_subdirs = count_target_directories_and_subdirs(directory)

            jacoco_files = find_jacoco_files(directory)
            if jacoco_files:
                if target_counts == 1:
                    print('\nProjetos com src na raiz\n')
                    for target_path, jacoco_file in jacoco_files:
                        origin_folder_name = os.path.basename(os.path.dirname(target_path))
                        destin_folder = os.path.join(file_destination, origin_folder_name.split('_')[0])
                        os.makedirs(destin_folder, exist_ok=True)
                        destin_folder_name = os.path.basename(destin_folder)
                        file_name = os.path.join(destin_folder, f"{destin_folder_name}_jacoco.csv")
                        # file_name = os.path.join(destin_folder, "jacoco.csv")
                        shutil.copy2(jacoco_file, file_name)
                    aggregate(destin_folder)
                    print(f"Jacoco '.csv' files copied to: {file_destination}")

                elif target_counts > 1:
                    print('\nMore than a file by folder...\n')

                    for project in os.listdir(directory):
                        destin_folder = os.path.join(file_destination, refactored_project.split('_')[0])
                        os.makedirs(destin_folder, exist_ok=True)
                        if os.path.isdir(os.path.join(directory, project)) and not project.startswith('.'):
                            jacoco_files = find_jacoco_files(os.path.join(directory, project))
                            for target_path, jacoco_file in jacoco_files:
                                origin_folder_name = os.path.basename(os.path.dirname(target_path))
                                file_name = os.path.join(destin_folder, f"{origin_folder_name}_jacoco.csv")
                                # file_name = os.path.join(destin_folder, "jacoco.csv")

                                shutil.copy2(jacoco_file, file_name)

                            print(f" Jacoco '.csv' files copied to: {file_destination}")
                    aggregate(destin_folder)

                else:
                    print('\n Target Count less than one. Verify!!!')
            else:
                print("File 'jacoco.csv' not found!!!")


if __name__ == "__main__":

    language = list_folders_to_menu(parent_dir, word='Language')

    destination_directory = f'{language}/coverage/cov_tool_report'
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    organize_dataset = input("\nDo you want to organize the dataset?(y/n)\n")
    if organize_dataset == 'y':
        dataset_management(language)

    else:
        data_base = list_folders_to_menu(os.path.join(parent_dir, language, 'dataset'), word='Data Base')
        dataset = get_split_name(data_base, '/', -1)
        if dataset == 'original':
            project = list_folders_to_menu(data_base, word='Project')
            project_name = get_split_name(project, '/', -1)
            original_destination_directory = os.path.join(destination_directory, '_dataset_original')
            if get_split_name(language, '/', -1) == 'java':
                print('Java')
                copy_original_jacoco_files(project, original_destination_directory)
            elif get_split_name(language, '/', -1) == 'python':
                print('Python')
                copy_html_files(project, original_destination_directory)

        else:
            approach_name = list_folders_to_menu(data_base, word='Approach')
            llm_model = list_folders_to_menu(approach_name, word='LLM')
            version_number = list_folders_to_menu(llm_model, word='Version')

            # project = list_folders_2_menu(version_number, word='Project')

            projects = os.listdir(version_number)

            for project in projects:
                project_file_path = os.path.join(version_number, project)
                print(project)


                approach: str = get_split_name(approach_name, '/', -1)
                llm: str = get_split_name(llm_model, '/', -1)
                version: str = get_split_name(version_number, '/', -1)
                # project_name: str = get_split_name(project, '/', -1)

                refactored_destination_directory = os.path.join(destination_directory, approach, llm, version, project)

                if language.split('/')[-1] == 'java':
                    copy_jacoco_files(project_file_path, refactored_destination_directory)

                elif language.split('/')[-1] == 'python':
                    copy_html_files(project_file_path, refactored_destination_directory)

                    # copy_jacoco_files(project_file_path, refactored_destination_directory)