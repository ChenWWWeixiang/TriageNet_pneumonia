import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
data=np.load('../../re/mosmed.npy')
y_pred=np.array(data[:,1:-1],np.float)
y_pred=np.argmax(y_pred,1)
y_true = np.array(data[:,-1],np.uint8)
sns.set()
f,ax=plt.subplots()
import csv
with open("go.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(y_pred.shape[0]):
        writer.writerow([y_pred[i],y_true[i]])
        #writer.writerow(y_true)

C2= confusion_matrix(y_true, y_pred,labels=[0,1],normalize='true')
print(C2) #
sns.heatmap(C2,annot=True,cmap='Blues',fmt="f")

ax.set_title('Confusion Matrix')
ax.set_xlabel('Prediction')
ax.set_ylabel('Ground Truth')
plt.xticks((0.5,1.5),['Non-pneumonia','COVID-19'])
plt.yticks((0.15,1.3),['Non-pneumonia','COVID-19'])
plt.savefig('../jpgs/confusionmatrix_mosmed.jpg')
plt.show()
