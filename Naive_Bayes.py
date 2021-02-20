import csv
import numpy as np
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
import os
company = 'boeing'
os.path.join(os.getcwd(),'Data','label_'+company+'.tsv')

data=[]

with open('Data/label_'+company+'.tsv', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        data.append(row)
tsvfile.close()
data=np.array(data)
print(np.shape(data))
#%%
posdata = []
negdata = []
for i in range(len(data)):
    if(data[i][3]=='0'):
        negdata.append(data[i][:3])
    else:
        posdata.append(data[i][:3])
posdata = np.array(posdata)
negdata = np.array(negdata)

nP, nN = len(posdata), len(negdata)

#%%
vectorizer = text.CountVectorizer()
X = vectorizer.fit_transform(data[:,1]).toarray()

words=vectorizer.vocabulary_

XN, XP = X[:nN,:], X[nN:,:]


Nk = np.zeros(2);
Nk[0] =  posdata.shape[0];
Nk[1] =  negdata.shape[0];
Pik = Nk/np.sum(Nk);

I=np.size(XN,0);
L=np.size(XP,0);
J = L + I

#%%
Njl_k0 = np.zeros((L+1,J));



for j in range(J):
    for i in range(I):
        for l in range(L+1):
                if(XN[i,j] == l):
                    Njl_k0[l,j] = Njl_k0[l,j] + 1;

muj_k0 = np.zeros(J);

s = np.zeros(J);

for j in range(J):
    for l in range(L + 1):
        s[j] = s[j] + l * Njl_k0[l, j];

for j in range(J):
    muj_k0[j] = (1 + s[j]) / (J + np.sum(s))

I=XP.shape[0];

Njl_k1 = np.zeros((L+1,J));


for j in range(J):
    for i in range(I):
        for l in range(L+1):
                if(XP[i,j] == l):
                    Njl_k1[l,j] = Njl_k1[l,j] + 1;

muj_k1 = np.zeros(J);

s = np.zeros(J);

for j in range(J):
    for l in range(L+1):
        s[j]= s[j] + l * Njl_k1[l,j];

sum1=0;
for j in range(J):
    muj_k1[j] = (1+s[j])/(J+np.sum(s))

pty = []
c = 0

headlines = data[:,1]

XX = vectorizer.transform(headlines).toarray()
for xd in XX:  # Iterate over disputed documents
    # Compare classes probabilities
    sum0 = 0;
    sum1 = 0;
    for j in range(J):
        sum0 = sum0 + xd[j] * np.log(muj_k0[j])
        sum1 = sum1 + xd[j] * np.log(muj_k1[j])

    pN = np.log(Pik[0]) + sum0
    pP = np.log(Pik[1]) + sum1
    pty.append([pN,pP,data[c,2]])
    c = c + 1

pn = np.array([p[0] for p in pty])
pp = np.array([p[1] for p in pty])
ps = np.array([float(p[2]) for p in pty])


pn = np.interp(pn, (pn.min(), pn.max()), (0, +1))
pp = np.interp(pp, (pp.min(), pp.max()), (0, +1))
ps = np.interp(ps, (pp.min(), pp.max()), (-1, +1))

label=[]
for i in range(len(pp)):
    if(pn[i]>pp[i]):
        label.append(0);
    else:
        label.append(1);

cn = pn * ps
cp = pp * ps
#n=[]
#p=[]

c = []
for i in range(len(cn)):
    if(label[i]==0):
        if(data[i,3]=='0'):
            c.append(-cn[i])
        else:
            c.append(-cn[i])
    else:
        if (data[i, 3] == '0'):
            c.append(cp[i])
        else:
            c.append(cp[i])


for i in range(len(data)):
    print(data[i,1], data[i,2], data[i,3], c[i])
#%%
with open(company + "_correlation.tsv", "w", newline='',encoding='utf-8') as writeFile:
        row = "Date" + "\t" + "Headline" + "\t" + "Stock Difference" + "\t" + "True Label" + "\t" + "Predicted Label" + "\t" + "Correlation" + "\n"
        writeFile.write(row)
        print(row)
        for i in range(len(data)):
                row = str(data[i,0]) +"\t" + str(data[i,1]) + "\t" + str(data[i,2]) + "\t" + str(data[i,3]) + "\t" + str(label[i]) + "\t" + str(c[i]) + "\n"
                writeFile.write(row)
                print(row)
            # Driver Code
writeFile.close()
#%%
np.savetxt(company + '_mujk.out', (muj_k0,muj_k1))
np.savetxt(company + '_Pik.out', Pik)