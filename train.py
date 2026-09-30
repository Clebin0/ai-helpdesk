from ml import train

if __name__=="__main__":
    _,metrics,comparison,_=train(save=True)
    print("selected_model:",metrics["selected_model"])
    print("macro_f1:",metrics["macro_f1"])
    for row in comparison:
        print(row)
