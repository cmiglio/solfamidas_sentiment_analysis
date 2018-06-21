import requests
import math
from bs4 import BeautifulSoup
import re
import unicodedata
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC 
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem import SnowballStemmer 
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import numpy as np
import scipy.sparse as ssp
from sklearn import preprocessing as prep
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import datetime
import warnings
warnings.filterwarnings('ignore')

def obtain_emos(emo):
    # Función para obtener el código de los emoticonos
    urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+\D\w+\D\w+\D\w+\D\w+', emo)

    codes=[]
    for url in urls:    
        aux=url.split('/')
        if aux[3] == 'emoji':
            codes.append(aux[-1])               

    return(codes)


def scrap_data_and_preprocess(data_submission):
    # Función para preprocesar los datos del fichero submission 
    # Leer aerolineas de fichero excel
    air=pd.read_excel('Listado aerolíneas.xlsx')
    aerolineas=air['airlines'].values.tolist()

    # Agrupar tweets por uno de los campos 

    df_new = data_submission.filter(['airline_sentiment'], axis=1)
    df_new["whole_tweet"] = ""
    df_new['emoticon']=""
    df_new['mentions']=""
    df_new['hashtags']=""
    df_new['airline']=""
    df_new['n_words']=0
    df_new['n_words_mayus']=0
    df_new['n_words_mayus_len']=0
    df_new['n_suspensivos']=0
    df_new['n_exclamations']=0
    df_new['n_questions']=0


    show_elap=100
    it=0
    n_tweets=data_submission.count().text
    ini=time.time()

    for indice_fila, fila in df_new.iterrows():
        it=it+1
        url=u'https://twitter.com/i/web/status/' + str(indice_fila)
        r = requests.get(url)
        soup = BeautifulSoup(r.text,'html.parser')
       
        # Extraer mediante scrapping la parte de la web que contiene el texto de los twits:
        hashtags = []
        mentions = []
        airlines = []

        try:
            # Extraer solamente el texto.
            tweet_text=[p.text for p in soup.findAll('p',class_='TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text')]
            text=str(tweet_text) 
            emo=str(soup.findAll('p',class_='TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text')[0])
            codes=obtain_emos(emo)
        except IndexError:
            tweet_text=str(data_submission.get_value(indice_fila, 'text'))
            text=tweet_text
            codes=[]

        # Eliminar hashtags, arrobas del tweet. guardar esta info en 'mentions y hashtags'. Eliminar webs
        word_list = text.split()

        # Eliminar caracteres del inicio y final del tweet
        pal=word_list[0]
        word_list[0]=pal[2:]
        pal=word_list[len(word_list)-1]
        word_list[len(word_list)-1]=pal[0:len(pal)-2]

        i = 0
        nwords=len(word_list)
        nwords_mayus=0
        nwords_mayus_len=0
        rem_list=[]
        while i < len(word_list):

            # Mirar si la palabra es mayuscula
            if word_list[i].isupper():
                nwords_mayus=nwords_mayus+1
                nwords_mayus_len=nwords_mayus_len+len(word_list[i])
            # Mirar si empieza por arroba
            if word_list[i].startswith('@'):
                aux=word_list[i]
                word_list[i]=aux[1:]  
                mentions.append(word_list[i])

            elif word_list[i].startswith('#'): 
                aux=word_list[i]
                word_list[i]=aux[1:]
                hashtags.append(word_list[i])        

            # Mirar si está en aerolinea
            wlist=word_list[i]   
            wlist=re.sub('[^A-Za-z0-9]+','',wlist)

            if wlist.lower() in aerolineas:
                airlines.append(wlist.lower())

            if 'http' in word_list[i]:
                n=word_list[i].find('http')
                aux=word_list[i]
                if n:
                    word_list[i]=aux[0:n]
                else:
                    rem_list.append(i)

            i+=1

        rem_list.sort(reverse=True)    
        for i in rem_list:
            word_list.remove(word_list[i])        

        text = ' '.join(word_list)


        n_suspensivos=text.count('...')
        n_exclamations=text.count('!')
        n_questions=text.count('?')

        text=text.lower()
        # Elimino acentos
        text = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii")
        # Elimino caracteres especiales
        text=re.sub('[^A-Za-z0-9]+',' ',text)

        df_new.set_value(indice_fila, 'whole_tweet', text, takeable=False)
        df_new.set_value(indice_fila, 'mentions', mentions, takeable=False)
        df_new.set_value(indice_fila, 'hashtags', hashtags, takeable=False)
        df_new.set_value(indice_fila, 'emoticon', codes, takeable=False)
        df_new.set_value(indice_fila, 'airline', airlines, takeable=False)
        df_new.set_value(indice_fila, 'n_words', nwords, takeable=False)
        df_new.set_value(indice_fila, 'n_words_mayus', nwords_mayus, takeable=False)
        df_new.set_value(indice_fila, 'n_words_mayus_len', nwords_mayus_len, takeable=False)
        df_new.set_value(indice_fila, 'n_suspensivos', n_suspensivos, takeable=False)
        df_new.set_value(indice_fila, 'n_exclamations', n_exclamations, takeable=False)
        df_new.set_value(indice_fila, 'n_questions', n_questions, takeable=False)


    # Iterar para saber cuanto tiempo falta (informativo solamente)
        if it % show_elap == 0:
            fin=time.time()
            print('Remaining time (estimated): ' + str((fin-ini) * (n_tweets-it) / it))


    # Aplicar una función (usando apply) para obtener un nuevo campo
    # En este caso aplicamos una función para calcular la longitud de texto

    df_new['len_tw']=df_new['whole_tweet'].apply(len)

    return df_new


def preprocess_data(data,data_raw,extra_features,add_stemmer,is_submission):
    if extra_features:
        data['n_words']=data['n_words'].fillna(0)
        data['n_words_mayus']=data['n_words_mayus'].fillna(0)
        data['n_words_mayus_len']=data['n_words_mayus_len'].fillna(0)
        data['len_tw']=data['len_tw'].fillna(0)
        data['n_suspensivos']=data['n_suspensivos'].fillna(0)
        data['n_exclamations']=data['n_exclamations'].fillna(0)
        data['n_questions']=data['n_questions'].fillna(0)
        data['n_emo']=data['emoticon'].apply(len)
        data['n_mentions']=data['mentions'].apply(len)
        data['n_hashtags']=data['hashtags'].apply(len)
        data['n_airline']=data['airline'].apply(len)

        data['week_day'] = data_raw['tweet_created'].str[0:3]
        data['month'] = data_raw['tweet_created'].str[4:7]
        data['day'] = pd.to_numeric(data_raw['tweet_created'].str[8:10])
        data['hour'] = pd.to_numeric(data_raw['tweet_created'].str[11:13])

        intervals = np.linspace(0, 24, 8, endpoint=False)
        data['hour_interval'] = np.digitize(data['hour'], intervals) - 1

        intervals = np.linspace(0, 31, 4, endpoint=False)
        data['day_interval'] = np.digitize(data['day'], intervals) - 1
        
        data=pd.concat([data, pd.get_dummies(data['day_interval'],prefix='dinterval')], axis=1)
        data=pd.concat([data, pd.get_dummies(data['hour_interval'],prefix='hinterval')], axis=1)
        data=pd.concat([data, pd.get_dummies(data['week_day'])], axis=1)
        
        data['n_words10']=data['n_words'].apply(lambda x: int(bool(x<=10)))
        data['n_words20']=data['n_words'].apply(lambda x: int(bool(x>10 and x<=20)))
        
        data['mayus_si']=data['n_words_mayus'].apply(lambda x: int(bool(x>0)))
        data['mayus_si1']=data['n_words_mayus'].apply(lambda x: int(bool(x==1)))
        data['mayus_si2mas']=data['n_words_mayus'].apply(lambda x: int(bool(x>1)))
        
        data['tweet_created'] = pd.to_datetime(data_raw.tweet_created)
# CREAMOS NUEVAS COLUMNAS PARA EL DÍA DE LA SEMANA Y LA HORA DEL DÍA
        data['weekday'] = data['tweet_created'].dt.weekday
        data['dayhour'] = data['tweet_created'].dt.hour
        data['finde']=np.where(data['weekday']<5,0,1)
        data['lun_mar']=np.where(data['weekday']<2,1,0)
        data['jue_vie']=data['weekday'].apply(lambda x: int(bool(x>2 and x<5)))
        data['time_negat']=data['dayhour'].apply(lambda x: int(bool(x>=18 and x<22 or x>=6 and x<9)))
        data['noche']=data['dayhour'].apply(lambda x: int(bool(x>=22 or x<6)))
        data['time_labor']=data['dayhour'].apply(lambda x: int(bool(x>=9 and x<18)))
        data['exclamations_si']=np.where(data['n_exclamations']>0,1,0)
        data['questions_si']=np.where(data['n_questions']>0,1,0)
        data['suspensivos_si']=np.where(data['n_suspensivos']>0,1,0)

# EMOTICONOS        
#        
        data['n_posit_emo']=0
        data['n_negat_emo']=0
        data['n_neutr_emo']=0

        emo_pol=pd.read_excel('Lista y polaridad emojis mejorado.xlsx',dtype={'Unicode_limpio':'str'})
        emo_pol['Unicode_limpio']=emo_pol['Unicode_limpio'].str.lower()
        from ast import literal_eval
        data.loc[:,'emoticon_reformat']=data.loc[:,'emoticon'].apply(lambda x: literal_eval(x))
        for i in data.index:
            for emot in data['emoticon_reformat'][i]:
                for j in range(len(emo_pol.Unicode_limpio)):
                    if emot == emo_pol.Unicode_limpio[j]:
                        if emo_pol.sentiment[j] == 'positive':
                            data.set_value(i,'n_posit_emo',data['n_posit_emo'][i] + 1,takeable=False)
                        if emo_pol.sentiment[j] == 'negative':
                            data.set_value(i,'n_negat_emo',data['n_negat_emo'][i] + 1,takeable=False)
                        if emo_pol.sentiment[j] == 'neutral':
                            data.set_value(i,'n_neutr_emo',data['n_neutr_emo'][i] + 1,takeable=False)

        data['emo_si']=data['emoticon_reformat'].apply(lambda d:int(bool(len(d)>0)))
        data['emo_pos_si']=data['n_posit_emo'].apply(lambda x: int(bool(x>0)))
# Userzone
        data['user_timezone']=data_raw.user_timezone.replace(np.nan, 'No_zone')
        data['user_timezone']=pd.Categorical(data_raw.user_timezone)        
        
        zonas = pd.read_excel('Zonas.xlsx')
        data['user_big_zone'] = 'nothing'

        for i in data_raw.index:
            for j in range(len(zonas.USER_ZONE)-1):
                if data_raw.user_timezone[i] == zonas.USER_ZONE[j]:
                    data.set_value(i,'user_big_zone',zonas.BIG_ZONE[j],takeable=False)
                else:
                    pass
        data['LatinAM_bool']=data['user_big_zone'].apply(lambda d:d=='RESTO_AM')
        
    
        data['iberia'] = 0
        data['ryanair'] = 0
        data['vueling'] = 0
        data['otras_cias'] = 0
        data['compania']=''

        for indice_fila, fila in data.iterrows():
            airlines1=data.get_value(indice_fila,'airline').split()
            long=len(airlines1)
            for cia in airlines1:     
                cia=re.sub('[^A-Za-z0-9]+','',cia)
                if cia =='iberia' or cia=='iberiaexpress':
                    data.set_value(indice_fila,'iberia',1,takeable=False)
                    if long==1:
                        data.set_value(indice_fila,'compania','iberia',takeable=False)
                elif cia =='vueling':
                    data.set_value(indice_fila,'vueling',1,takeable=False)
                    if long==1:
                        data.set_value(indice_fila,'compania','vueling',takeable=False)
                elif cia =='ryanair':
                    data.set_value(indice_fila,'ryanair',1,takeable=False)
                    if long==1:
                        data.set_value(indice_fila,'compania','ryanair',takeable=False)
                elif cia=='':
                    cia=''
                else:
                    data.set_value(indice_fila,'otras_cias',1,takeable=False)
                    if long==1:
                        data.set_value(indice_fila,'compania','otras',takeable=False)

    # Cambiar tu destino a un solo click
    
        if is_submission==False:
            d=data['hashtags'].apply(lambda x: bool(re.match(r'.*(Hola[A-Z]{1})[a-z]+.*',x)))
            data.airline_sentiment[d==True]='neutral'
            
        data['is_reply']=data_raw.is_reply.astype(int)

        
        data['mentions_si']=data['n_mentions'].apply(lambda x: int(bool(x>2)))
        data['hashtags_si']=data['n_hashtags'].apply(lambda x: int(bool(x>2)))
        
        data['n_hashtags9']=data['n_hashtags'].apply(lambda x: int(bool(x>2 and x<=9)))
        data['n_hashtags23']=data['n_hashtags'].apply(lambda x: int(bool(x>9 and x<=23)))
        data['n_hashtags24mas']=data['n_hashtags'].apply(lambda x: int(bool(x>23)))
        
        data['spanair']=data['whole_tweet'].apply(lambda x: int(bool('accidente' in x and 'spanair' in x)))


    tt = TweetTokenizer() # defino el tokenizer
    snowball_stemmer = SnowballStemmer('spanish') # defino el stemmer en ingles
    data['stemmed']=data['whole_tweet'] # Creo una nueva columna llamada stemmer con los datos de los tweets
    
    if add_stemmer:
        # tokenizo (divido por palabras)
        data['stemmed']= data['stemmed'].apply(tt.tokenize) 
        # hago el stemmer (elimino lexemas)
        data['stemmed'] = data['stemmed'].apply(lambda x: [snowball_stemmer.stem(y) for y in x]) 
        # Vuelvo a juntar en strings
        data['stemmed'] = data['stemmed'].apply(lambda x: ' '.join(x)) 

    # juntar emoticonos con los datos stemmed: PARECE QUE AÑADIR LOS EMOJIS DENTRO DEL TEXTO NO MEJORA
        #data['stemmed'] = data[['stemmed', 'emoticon']].apply(lambda x: ' '.join(x), axis=1)
   
    return data

def obtain_data_representation(df, test,max_df,binary,max_features,
                               ngram_range,norm,extra_features,add_stemmer,
                               add_stopwords):
    # If there is no test data, split the input
    if test is None:
        # Divide data in train and test
        train, test = train_test_split(df, test_size=0.25)
        df.airline_sentiment = pd.Categorical(df.airline_sentiment)
    else:
        # Otherwise, all is train
        train = df
        
    if max_features == 0:
        max_features=None
    
    if add_stopwords:
        sw_df=pd.read_csv('stopwords.csv')
        sw=sw_df['stopwords_token'].values.tolist()
    else:
        sw=None    
    
                     
    # Create a Bag of Words (BoW), by using train data only. Mi aportación aqui es quitar los acentos y poner
    # el parámetro max_df a 0.2 (en base a prueba y error) , quitando el number of features también mejora
    cv = CountVectorizer(strip_accents='unicode',
                         max_df=max_df,
                         binary=binary,
                         max_features=max_features,
                         ngram_range=(1,ngram_range),
                         stop_words=sw)
    
    x_train = cv.fit_transform(train['stemmed'])
    y_train = train['airline_sentiment'].values
    
    # Normalizo las features para que no afecte la longitud del twit
    if norm:
        tf_transformer=TfidfTransformer(use_idf=False).fit(x_train)
        x_train=tf_transformer.transform(x_train) 
    
    # Obtain BoW for the test data, using the previously fitted one
    x_test = cv.transform(test['stemmed'])
    if norm:
        tf_transformer=TfidfTransformer(use_idf=False).fit(x_test)
        x_test=tf_transformer.transform(x_test)
    
    # Extracción de features 'extra'
    if extra_features:
        
        features = ['exclamations_si','questions_si','suspensivos_si',
                    'mentions_si',A'n_posit_emo',
                    'iberia','vueling','ryanair','otras_cias',
                    'n_words10','n_words20','LatinAM_bool',
                    'is_reply','lun_mar','finde','noche','time_negat',
                    'hashtags_si','mayus_si']
        # Training
        x_train_2 = train.loc[:, features].values
        x_train_2 = prep.scale(x_train_2)
        x_train_2 = np.asmatrix(x_train_2,dtype=np.float64)
        # Test
        x_test_2 = test.loc[:, features].values
        x_test_2 = prep.scale(x_test_2)
        x_test_2 = np.asmatrix(x_test_2,dtype=np.float64)
       
        # Juntar ambas matrices
        x_train=np.concatenate((x_train_2,x_train.todense()),axis=1)
        x_test=np.concatenate((x_test_2,x_test.todense()),axis=1)
        
        x_train=ssp.csr_matrix(x_train)
        x_test=ssp.csr_matrix(x_test)
    
    try:
        y_test = test['airline_sentiment'].values
    except:
        # It might be the submision file, where we don't have target values
        y_test = None
        
    return {
        'train': {
            'x': x_train,
            'y': y_train
        },
        'test': {
            'x': x_test,
            'y': y_test
        }
    }

def train_model(dataset, dmodel, *model_args, **model_kwargs):
    # Create a Naive Bayes model
    model = dmodel(*model_args, **model_kwargs)
    
    # Train it
    model.fit(dataset['train']['x'], dataset['train']['y'])
    
    # Predict new values for test
    y_pred = model.predict(dataset['test']['x'])
    
    # Print accuracy score unless its the submission dataset
    confu = None
    gab = None
    bag = None
    if dataset['test']['y'] is not None:
        score = accuracy_score(dataset['test']['y'], y_pred)
        confu = confusion_matrix(dataset['test']['y'], y_pred, labels=['positive', 'neutral', 'negative'])
        print("Model score is: {}".format(score))
        
        gab = [True if dataset['test']['y'][x]=='positive' and y_pred[x]=='negative' else False for x in range(len(y_pred))]
        bag = [True if dataset['test']['y'][x]=='negative' and y_pred[x]=='positive' else False for x in range(len(y_pred))]

    # Done
    return model, y_pred, confu, gab, bag


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
def create_submit_file(df_submission, ypred):
    date = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    filename = 'submission_' + date + '.csv'
    
    df_submission['airline_sentiment'] = ypred
    df_submission[['airline_sentiment']].to_csv(filename)
    
    print('Submission file created: {}'.format(filename))
    print('Upload it to Kaggle InClass')