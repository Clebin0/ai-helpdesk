from ml import train

if __name__=="__main__":
    _,metrics=train(save=True)
    print(metrics)
