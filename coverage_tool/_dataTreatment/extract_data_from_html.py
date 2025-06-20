import pandas as pd
import os
import shutil
from bs4 import BeautifulSoup
from _dataTreatment.models.curingas import list_folders_to_menu, get_split_name

ROOT = os.getcwd()
parent_dir = os.path.dirname(ROOT)


def get_data_from_html(html_file_path) -> list:
    """
    function to extract tables from html file
    :param html_file_path:
    :return:
    """
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    tables = soup.find_all('table')
    data_frames = []
    for table in tables:
        table_data = []
        rows = table.find_all('tr')
        for row in rows:
            # Find all cells in a line
            cells = row.find_all(['td', 'th'])
            cell_data = [cell.get_text(strip=True) for cell in cells]
            table_data.append(cell_data)
        table_df = pd.DataFrame(table_data)
        data_frames.append(table_df)
    return data_frames


def save_tables_to_csv(data_table, csv_output_file):
    """
    function to save tables to csv
    :param data_table:
    :param csv_output_file:
    :return:
    """
    all_tables_df = pd.concat(data_table, ignore_index=True)
    all_tables_df.to_csv(csv_output_file, index=False, header=False)


if __name__ == "__main__":
    """
    This script moves the index.html files from the htmlcov folder to a folder in coverage/cov_tool_report/(LLM) 
    with the same name as the analyzed project and then moves the index.html inside.
    """
    html_file = 'index.html'
    jacoco_file = 'jacoco.csv'

    flag = int(input('\nYou want to generate original files?\n Yes - 1\n No - 2\n'))

    language = list_folders_to_menu(parent_dir, word='Language')
    version = []
    llm = []
    project_name = []
    smell_name = []

    if flag == 1:
        project_name = os.path.join(language, 'coverage/cov_tool_report/_dataset_original/')
        project_folders = [folder for folder in os.listdir(project_name)]

        if language.split('/')[-1] == 'python':
            for folder in project_folders:
                html_path = os.path.join(project_name, folder, html_file)
                tables_data = get_data_from_html(html_path)
                path = f'{language}/coverage/extraction_results/_dataset_original/'
                if not os.path.exists(path):
                    os.makedirs(path)
                output_csv = f'{path}/(Original_File)-{folder}.csv'
                save_tables_to_csv(tables_data, output_csv)
                print(f'Generating original {folder} (.csv) coverage report...')

        elif language.split('/')[-1] == 'java':
            for folder in project_folders:
                project_path = os.path.join(project_name, folder, jacoco_file)
                path = f'{language}/coverage/extraction_results/_dataset_original/'
                if not os.path.exists(path):
                    os.makedirs(path)
                output_csv = f'{path}/(Original_File)-{folder}.csv'
                shutil.copy(project_path, output_csv)
                print(f'Generating original {folder} (.csv) coverage report...')

            print('Done!!!')

    elif flag == 2:

        approach = list_folders_to_menu(os.path.join(language, 'coverage/cov_tool_report'), word='Approach')
        llm = list_folders_to_menu(approach, word='LLM')
        version = list_folders_to_menu(llm, word='Version')

        projects = os.listdir(version)
        if language.split('/')[-1] == 'python':

            for project in projects:
                smells_folders = [folder for folder in os.listdir(os.path.join(version, project))]
                for smell in smells_folders:
                    html_path = os.path.join(version, project, smell, html_file)
                    tables_data = get_data_from_html(html_path)

                    llm = get_split_name(llm, '/', -1)
                    approach = get_split_name(approach, '/', -1)
                    version_number = get_split_name(version, '/', -1)

                    path = f'{language}/coverage/extraction_results/{approach}/{llm}/{version_number}/{project}/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    output_csv_file = f'{path}/{project}.{llm}.{version_number}.{smell}.csv'
                    save_tables_to_csv(tables_data, output_csv_file)
                    print(f'Generating {smell} (.csv) report...')

        elif language.split('/')[-1] == 'java':

            for project in projects:
                smells_folders = [folder for folder in os.listdir(os.path.join(version, project))]
                for smell in smells_folders:
                    project_path = os.path.join(version, project, smell, jacoco_file)
                    # tables_data = get_data_from_html(html_path)

                    llm = llm.split('/')[-1] # get_split_name(llm, '/', -1)
                    approach = approach.split('/')[-1] #get_split_name(approach, '/', -1)
                    version_number = version.split('/')[-1] # get_split_name(version, '/', -1)

                    path = f'{language}/coverage/extraction_results/{approach}/{llm}/{version_number}/{project}/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    output_csv_file = f'{path}/{project}.{llm}.{version_number}.{smell}.csv'
                    shutil.copy(project_path, output_csv_file)

                    # save_tables_to_csv(tables_data, output_csv_file)
                    print(f'Generating {smell} (.csv) report...')
                    

        print('Done!!!')
