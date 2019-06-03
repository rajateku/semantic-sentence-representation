import warnings
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

warnings.filterwarnings(action='ignore')

tf.logging.set_verbosity(tf.logging.ERROR)
gloveFile = "/home/user/glove_pretrained_models/glove.6B.50d.txt"

def load_pretrained_word2vec_model():
    f2 = open(gloveFile, 'r')
    model = {}
    for line in f2:
        splitline = line.split()
        word = splitline[0]
        embedding = [float(val) for val in splitline[1:]]
        model[word] = embedding
    return model

pretrained = load_pretrained_word2vec_model()

text = '''
collected a band of young men and trained them in warfare. They lived in a forest hideout on the banks of the river Tungabhadra in South India.

One day, the brothers were out on a hunt. Ferocious dogs accompanied them. They crossed the river and rode on. A couple of frightened rabbits ran out of the bushes. The dogs gave them chase with the two brothers closely behind on their horses.

It was a long chase. The rabbits were running for their life. The dogs were catching up. Suddenly, in a swift move, the rabbits turned and faced the dogs. Taken aback by the show of defiance, the barking dogs stepped back. Hakka called back the dogs. As the dogs turned back, the rabbits walked away.

Hakka looked around. They were on the other side of the Tungabhadra. It was a rocky land. The sun was blazing in the sky.

“Strange! I’ve never seen rabbits challenging dogs before!” said Bukka.

“That’s the quality of this land,” said a quiet voice, “Even rabbits give fight.”

Startled to hear a stranger speak, the two brothers turned.

They saw a holy man walking towards them. He was a picture of peace. At the same time, his eyes were blazing bright
'''

f = open("tensorboard_word_embeddings/golden_english_sentences.txt", "r")
text = f.read()


text = text.replace("," , "")
text = text.replace("." , "")
text = text.replace("!" , "")
text = text.replace("-" , " ")
text = text.lower()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = text

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
word_tokens = set(word_tokens)

print(word_tokens)




word_list = word_tokens
max_size = len(word_list)
w2v = np.zeros((max_size, 50))

if not os.path.exists('projections'):
    os.makedirs('projections')

with open("projections/metadata.tsv", 'w+') as file_metadata:
    for i, word in enumerate(word_list):
        # store the embeddings of the word
        try:
            w2v[i] = pretrained[word]
        except:
            w2v[i] = pretrained["a"]

        # write the word to a file
        file_metadata.write(word + '\n')

sess = tf.InteractiveSession()

with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable=False, name='embedding')

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter('projections', sess.graph)
config = projector.ProjectorConfig()
embed= config.embeddings.add()

embed.tensor_name = 'embedding'
embed.metadata_path = 'metadata.tsv'

projector.visualize_embeddings(writer, config)
saver.save(sess, 'projections/model.ckpt', global_step=max_size)


os.system("tensorboard --logdir=projections --port=8080")
