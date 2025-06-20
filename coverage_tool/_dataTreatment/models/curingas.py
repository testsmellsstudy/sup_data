import os
import shutil
import numpy as np
import re


def list_folders_to_menu(root, word):
    # folder recebe 'f' para \qualquer 'f' dentro da raiz, se o join da raiz + f for um diretório
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))
               and not f.startswith('.')
               and not f.startswith('_')]
    print('\n')
    # folder = int(input(f'\nWhich {word} do you want to do with?\n'))

    for i, pastas in enumerate(folders):
        print('{} - {}'.format(i, pastas))

    folder = int(input(f'\nWhich {word} do you want to do with?\n'))
    return os.path.join(root, folders[folder])


def list_files_2_menu(root, word):
    # folder recebe 'f' para \qualquer 'f' dentro da raiz, se o join da raiz + f for um diretório
    folders = [f for f in os.listdir(root) if not os.path.isdir(os.path.join(root, f))
               and not f.startswith('.')
               and not f.startswith('_')]

    print('\n')
    for i, pastas in enumerate(folders):
        print('{} - {}'.format(i, pastas))
    folder = int(input(f'\nWhich {word} do you want to do with?\n'))
    return os.path.join(root, folders[folder])


def organize_directories(base_path, language):
    """
        Função para percorrer um diretório pegar todos os subdiretórios e criar um novo
        diretório com o o valor de "int: broke" que forma o nome "base_name" e coloca
        todos os diretórios com o mesmo nome dentro de seu respectivo diretório.
    """
    # Get subdirectories list
    sub_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if language.split('/')[-1] == 'python':
        broke = 6
    elif language.split('/')[-1] == 'java':
        broke = 7

    sub_dirs_name = {}
    for subdir in sub_dirs:
        base_name = subdir.split('_')[broke]

        if base_name not in sub_dirs_name:
            sub_dirs_name[base_name] = []

        sub_dirs_name[base_name].append(subdir)

    # Create and allocate the new directories in your folders
    for base_name, sub_dirs_list in sub_dirs_name.items():
        new_directory = os.path.join(base_path, base_name)
        os.makedirs(new_directory, exist_ok=True)

        for subdir in sub_dirs_list:
            src_path = os.path.join(base_path, subdir)
            dest_path = os.path.join(new_directory, subdir)
            shutil.move(src_path, dest_path)


def number_split_words(texto):
    code = re.compile(r'[a-zA-Z]+')
    value = re.compile(r'\d+')
    code_1 = code.findall(texto)
    value_2 = value.findall(texto)

    return code_1[0], value_2[0]


def get_split_name(name, charc, split) -> str:
    """
    Function that brake the large string separated by a character and return the name chased
    :param charc:
    :param split:
    :param name:
    :return:
    """
    return name.split(f'{charc}')[split]


def sums_calculate(column):
    """
        Function to calculate the positives and negatives sums values of a column
    """
    total_sum = column.sum()
    positive_sum = column[column > 0].sum()
    negative_sum = column[column < 0].sum()
    return total_sum, positive_sum, negative_sum


def altera_nome():
    repo_name = 'aws-sam-cli'
    root = f'/home/none/laboratory/msc/smell.Mission#1/pytest/datasets/refactored_by_smell/{repo_name}'
    # Verifica se o diretório pai existe
    if os.path.exists(root) and os.path.isdir(root):
        # Lista todos os subdiretórios no diretório pai
        subdirectories = [os.path.join(root, nome) for nome in os.listdir(root) if
                          os.path.isdir(os.path.join(root, nome))]
        # Itera sobre os subdiretórios
        for subdirectory in subdirectories:
            # Separa o nome do subdiretório em palavras
            palavras = subdirectory.split(os.path.sep)[-1].split()
            # Renomeia o subdiretório intercalando as palavras com "_"
            novo_nome = "_".join(palavras)
            # Caminho completo para o novo nome do subdiretório
            new_path = os.path.join(root, novo_nome)
            # Renomeia o subdiretório
            os.rename(subdirectory, new_path)
            print(f"Renomeado {subdirectory} para {new_path}")
    else:
        print(f'O diretório pai {root} não existe ou não é um diretório.')


def replace_file():
    repo_name = 'aws-sam-cli'
    path = f'/home/none/laboratory/msc/smell.Mission#1/pytest/datasets/refactored_by_smell/{repo_name}/'
    repositories = np.asarray(os.listdir(path)).tolist()

    diretorio = ('/home/none/laboratory/msc/smell.Mission#1/pytest/datasets/refactored_by_smell/aws-sam-cli/aws-sam'
                 '-cli_original')
    file_name = 'pytest.ini'  # Nome do arquivo que você deseja substituir

    # Verifica se o diretório existe
    if os.path.exists(diretorio) and os.path.isdir(diretorio):
        caminho_arquivo_antigo = os.path.join(diretorio, file_name)

        # Verifica se o arquivo antigo existe
        if os.path.exists(caminho_arquivo_antigo):
            for repository in repositories:
                if repository == 'aws-sam-cli_original' or repository == 'venv':
                    pass
                else:
                    caminho_arquivo_novo = f'{path}{repository}/pytest.ini'  # Substitua pelo caminho do novo arquivo
                    shutil.copy(caminho_arquivo_antigo, caminho_arquivo_novo)
                    print(f'O arquivo {file_name} foi substituído com sucesso.')
        else:
            print(f'O arquivo {file_name} não foi encontrado no diretório.')
    else:
        print(f'O diretório {diretorio} não existe ou não é um diretório.')


def rename_files(repository, repo):
    syllables = repository.split('_')
    if len(repository) > 30:
        repository = '_'.join(syllables[2:])
        repo.append(repository)
    else:
        repository = '_'.join(syllables[1:])
        repo.append(repository)


def split_and_join_name(name):
    # Divide o nome pelos "_"
    parts = name.split("_")

    # Exibe as partes do nome com seus índices
    print("Partes do nome:")
    for i, part in enumerate(parts):
        print(f"{i + 1}: {part}")

    # Pergunta ao usuário quais partes deseja juntar
    choices = input("Digite os números das partes que deseja juntar separados por vírgula (ex: 1,4): ").split(",")

    # Verifica se as escolhas são válidas
    for choice in choices:
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(parts):
            return "Escolha inválida"

    # Junta as partes escolhidas
    new_name_parts = [parts[int(choice) - 1] for choice in choices]
    new_name = "_".join(new_name_parts)

    return new_name

    # # Nome de exemplo
    # nome = "vhs-teletext_ali1234_-_dataset_validation_-_14-01-2024_16-27-56_-_master_ref_GeneralFixture_00.06."
    #
    # # Divide e junta o nome
    # novo_nome = split_and_join_name(nome)
    # print("Novo nome:", novo_nome)
