import re
import nltk
from nltk.corpus import stopwords
import os
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.models import LsiModel
from gensim.models import TfidfModel
import logging
from gensim import similarities
from gensim.test.utils import datapath
import time
from gensim.similarities import docsim,Similarity
import heapq
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open
from bs4 import BeautifulSoup


BASE_TRAIN_FILE='recommendation'
MODEL_FILE=os.path.join(os.path.join(os.getcwd(), 'model'), 'lda_model')
INDEX_FILE=os.path.join(os.path.join(os.getcwd(), 'index'), 'index')
Dict_FILE=os.path.join(os.path.join(os.getcwd(), 'dict'), 'dictionary')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
OLD_FILE='old_file.txt'



def classfy_file_type(file_path):
    '''以list形式返回不同文件类型的路径'''
    pdf_name = []
    html_name = []
    for root, dirs, files in os.walk(file_path):
        for name in files:
            print(name)
            if re.search('\.pdf', name):
                pdf_name.append(os.path.join(root, name))
            else:
                html_name.append(os.path.join(root, name))
    return pdf_name ,html_name


def readPDF(file_path,file):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams(detect_vertical=True)
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    print(file)
    pdfFile=open(os.path.join(file_path,file), 'rb')
    try:
        process_pdf(rsrcmgr, device, pdfFile)
    except Exception as e:
        pass
        pdfFile.close()
        print(e)
        print('file:'+file_path+'has some problem and been passed')
        f=open('problems_file.txt')
        if file in f.read():
            print('file: ' + file + ' has been recorded  ')
        else:
          with open('problems_file.txt','a') as f:
              f.write(file + '\n')
              f.close()
    device.close()
    content = retstr.getvalue()
    retstr.close()
    return content

def read_html(file_path,file):
        file = os.path.join(file_path,file)
        htmlfile = open(file, 'r', encoding='utf-8')
        htmlpage = htmlfile.read()
        bs = BeautifulSoup(htmlpage, 'lxml')
        content = bs.getText()
        return content


def complet_text(txt):
    #通过txt获取文章的分词形式（去除了停止词）
  extract_words = re.sub(r'[^a-zA-Z]', ' ', txt)
  small_words = extract_words.lower()
  tokens = nltk.word_tokenize(small_words) #分词化
  tokens=filter_stop_words(tokens)
  return tokens


def filter_stop_words(tokens):
    filer_words = [word for word in tokens if word not in stopwords.words('english')] #去除停止词
    return filer_words


def get_diff_type_filetoken(file_path):
    std_docs = []#获取所有文件的分词文档库[[文档1],[文档2]]
    pdf_file_name=[]#pdf文件的名称
    html_file_name=[]#html文件的名称
    pdf_name, html_name = classfy_file_type(file_path)
    for pdf in pdf_name :
        txt1 = readPDF(os.path.dirname(pdf),os.path.basename(pdf))
        file_token = complet_text(txt1)
        pdf_file_name.append(os.path.basename(pdf))
        std_docs.append(file_token)
    for html in html_name :
        txt2 = read_html(os.path.dirname(html), os.path.basename(html))
        file_token = complet_text(txt2)
        html_file_name.append(os.path.basename(html))
        std_docs.append(file_token)
    file_name_list = pdf_file_name+html_file_name
    return file_name_list ,std_docs


def get_std_docs(BASE_TRAIN_FILE):
    std_docs=[]
    file_names_list=[]
    for file_name in os.listdir(BASE_TRAIN_FILE):
        file_path=os.path.join(BASE_TRAIN_FILE,file_name)
        txt = open(file_path, 'r',encoding='utf-8')
        txt=txt.read()
        file_token=complet_text(txt)
        std_docs.append(file_token)
        file_names_list.append(file_name)
    return file_names_list,std_docs

def get_std_docs_without_write(BASE_TRAIN_FILE):

    std_docs=[]
    file_names_list=[]
    for file_name in os.listdir(BASE_TRAIN_FILE):
        file_path=os.path.join(BASE_TRAIN_FILE,file_name)
        txt = open(file_path, 'r',encoding='utf-8')
        txt=txt.read()
        file_token=complet_text(txt)
        std_docs.append(file_token)
        file_names_list.append(file_name)
    return file_names_list,std_docs


def doc_prepare(new_file):
    "新文档待相识度计算"
    _,new_file_token=get_std_docs(new_file)
    new_file_dic=Dictionary(new_file_token)
    new_file_corpus=[new_file_dic.doc2bow(text) for text in new_file_token]
    return new_file_corpus

def lda_model(common_corpus,new_corpus):
    lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)
    temp_file = datapath("lad_model")
    lda.save(temp_file)
    lda = LdaModel.load(temp_file)
    other_corpus = [common_dictionary.doc2bow(text) for text in new_corpus]
    lda.update(other_corpus)




def get_old_file(OLD_FILE):
    with open(OLD_FILE, 'r') as f:
        old_file_list = f.read().splitlines()
        print(old_file_list)
    return old_file_list

def get_added_files(OLD_FILE):
    old_db_file=get_old_file(OLD_FILE)  # 已经训练过的老文章list
    print('old_files num :'+str(len(old_db_file)))
    file_name = os.listdir(BASE_TRAIN_FILE)# 最新文章集合的list
    print('total files num : '+str(len(file_name)))
    new_added_file_name = list(set(file_name).difference(set(old_db_file)))
    with open(OLD_FILE, 'a') as f:
        for i in new_added_file_name:
            f.write(i + '\n')
    print('new_added_file nums : '+str(len(new_added_file_name)))
    return new_added_file_name

def get_new_file_token(new_file_name):
    'new_file_name is a file list'
    train_file_path=BASE_TRAIN_FILE
    new_files_token=[]
    for file in new_file_name:
        file_path=os.path.join(train_file_path,file)
        txt = open(file_path, 'r',encoding='utf-8')
        txt=txt.read()
        file_token=complet_text(txt)
        new_files_token.append(file_token)
    print(new_files_token)
    return new_files_token

def update_model_and_index_dict():
    t1=time.time()
    new_file_name=get_added_files(OLD_FILE)
    new_files_token=get_new_file_token(new_file_name)

    #update dictionary
    common_dictionary = Dictionary.load_from_text(Dict_FILE)
    common_dictionary.add_documents(new_files_token)
    common_dictionary.save_as_text(Dict_FILE)

    #update model and saved
    file_names_list, std_docs = get_std_docs_without_write(BASE_TRAIN_FILE)#先使用全部的corpus对模型进行训练，后期可以考虑修改
    common_corpus = [common_dictionary.doc2bow(text) for text in std_docs]
    new_corpus = [common_dictionary.doc2bow(text) for text in new_files_token]
    # lda=LdaModel.load(MODEL_FILE)
    # lda.update(new_corpus)
    lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)
    lda.save(MODEL_FILE)

    #update index and saved
    index=Similarity.load(INDEX_FILE)
    index.add_documents(new_corpus)
    index.save(INDEX_FILE)
    print('finished complete the model index and dictionary')
    print('model update costs : '+ str(time.time()-t1))
    return new_file_name


def check_modify_time(base_file_path):
    return os.path.getmtime(base_file_path)

def check_update(base_file_path):
    t1=check_modify_time(base_file_path)
    time.sleep(10)
    t2=check_modify_time(base_file_path)
    if t1==t2:
        print('update dict index and model')
        new_files=update_model_and_index_dict()
        recommend_files=get_recommended_file()
    return recommend_files
def get_recommend():
    recommend_list=[]
    file_names_list,std_docs = get_std_docs(BASE_TRAIN_FILE)
    #判断是否创建字典存储目录
    if not os.path.exists(os.path.join(os.getcwd(), 'dict')):
        os.makedirs(os.path.join(os.getcwd(), 'dict'))
        common_dictionary = Dictionary(std_docs)
        common_dictionary.save_as_text(Dict_FILE)
    else:
        common_dictionary=Dictionary.load_from_text(Dict_FILE)
    common_corpus=[common_dictionary.doc2bow(text) for text in std_docs]
    # 判断是否创建模型存储目录
    if not os.path.exists(os.path.join(os.getcwd(), 'model')):
        os.makedirs(os.path.join(os.getcwd(), 'model'))
        lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)
        lda.save(MODEL_FILE)
    else:
        lda = LdaModel.load(MODEL_FILE)
    # 判断是否创建index存储目录
    if not os.path.exists(os.path.join(os.getcwd(), 'index')):
        os.makedirs(os.path.join(os.getcwd(), 'index'))
        index_file = os.path.join(os.path.join(os.getcwd(), 'index'), 'index')
        index = Similarity(index_file, common_corpus, num_features=len(common_dictionary))
        index.save(index_file)
        print('from here 0000')
    else :
        index_file = os.path.join(os.path.join(os.getcwd(), 'index'), 'index')
        index=Similarity.load(index_file)
        print('from here 111')

    # index.save('/tmp/deerwester.index')
    #>>> index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    # This is true for all similarity indexing classes (similarities.Similarity, similarities.MatrixSimilarity and similarities.SparseMatrixSimilarity)
    print(index)
    for similarities in index:
        print(similarities)
        a = similarities.tolist()
        recommend_list.append([a.index(key) for key in heapq.nlargest(10, a)])
    return file_names_list,recommend_list

def get_recommend2():
    recommend_list=[]
    file_names_list,std_docs = get_diff_type_filetoken('\\\\10.6.1.31\\article')
    #判断是否创建字典存储目录
    if not os.path.exists(os.path.join(os.getcwd(), 'dict')):
        os.makedirs(os.path.join(os.getcwd(), 'dict'))
        common_dictionary = Dictionary(std_docs)
        common_dictionary.save_as_text(Dict_FILE)
    else:
        common_dictionary=Dictionary.load_from_text(Dict_FILE)
    common_corpus=[common_dictionary.doc2bow(text) for text in std_docs]
    # 判断是否创建模型存储目录
    if not os.path.exists(os.path.join(os.getcwd(), 'model')):
        os.makedirs(os.path.join(os.getcwd(), 'model'))
        lda = LdaModel(common_corpus, num_topics=50, alpha='auto', eval_every=5)
        lda.save(MODEL_FILE)
    else:
        lda = LdaModel.load(MODEL_FILE)
    # 判断是否创建index存储目录
    if not os.path.exists(os.path.join(os.getcwd(), 'index')):
        os.makedirs(os.path.join(os.getcwd(), 'index'))
        index_file = os.path.join(os.path.join(os.getcwd(), 'index'), 'index')
        index = Similarity(index_file, common_corpus, num_features=len(common_dictionary))
        index.save(index_file)
        print('from here 0000')
    else :
        index_file = os.path.join(os.path.join(os.getcwd(), 'index'), 'index')
        index=Similarity.load(index_file)
        print('from here 111')
    # index.save('/tmp/deerwester.index')
    #>>> index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    # This is true for all similarity indexing classes (similarities.Similarity, similarities.MatrixSimilarity and similarities.SparseMatrixSimilarity)
    print(index)
    for similarities in index:
        print(similarities)
        a = similarities.tolist()
        recommend_list.append([a.index(key) for key in heapq.nlargest(10, a)])
    return file_names_list,recommend_list

def get_recommended_file():
    file_names,recommend_list=get_recommend()
    print(file_names)
    print(recommend_list)
    recommend=[[] for i in range(len(file_names))]
    for i, file in enumerate(file_names):
        for j in range(10):
            recommend[i].append(file_names[recommend_list[i][j]])
    print(recommend)
    return recommend
def get_recommended_file2():
    file_names,recommend_list=get_recommend2()
    print(file_names)
    print(recommend_list)
    recommend=[[] for i in range(len(file_names))]
    for i, file in enumerate(file_names):
        for j in range(10):
            recommend[i].append(file_names[recommend_list[i][j]])
    print(recommend)
    return recommend
if __name__ == '__main__':
    # recommend_list = get_recommended_file()
    get_recommended_file2()
