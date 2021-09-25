import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
data=np.load('../re/ind_try.npy')
y_pred=np.array(data[:,1:-1],np.float)
y_pred=np.argmax(y_pred,1)
y_true = np.array(data[:,-1],np.uint8)
sns.set()
f,ax=plt.subplots()
import csv
with open("go.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(y_pred)
    writer.writerow(y_true)

C2= confusion_matrix(y_true, y_pred,labels=[0,1,2],normalize='true')
print(C2) #
sns.heatmap(C2,annot=True,cmap='Blues',fmt="f")

ax.set_title('Confusion Matrix')
ax.set_xlabel('Prediction')
ax.set_ylabel('Ground Truth')
plt.xticks((0.4,1.5,2.5),['Normal','CP','COVID-19'])
plt.yticks((-0.2,1.4,2.2),['Normal','CP','COVID-19'])
plt.savefig('jpgs/confusionmatrix.jpg')
plt.show()
