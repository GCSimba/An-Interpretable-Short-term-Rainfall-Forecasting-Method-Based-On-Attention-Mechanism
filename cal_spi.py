try:
    import Tkinter as tk
    import tkMessageBox as message
    import  tkFileDialog as filedialog
    import ttk
except:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk
    from tkinter import messagebox as message

import pandas as pd

def autolabel(ax, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    
    """
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
#    for i in ax.patches:
#        # get_width pulls left or right; get_y pushes up or down
#        ax.text(i.get_width()+.3, i.get_y()+.1, \
#                str(round((i.get_width()/total)*100, 2))+'%\n n= '+str(total), fontsize=7,color='dimgrey')
##    xpos = xpos.lower()  # normalize the case of the parameter
##    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
##    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
##
##    for rect in rects:
##        height = rect.get_height()
##        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
##                '{}'.format(height), ha=ha[xpos], va='bottom')

        
SPIclasses=pd.DataFrame(data=['SPI ≤ -2', '-2 < SPI ≤ -1.5', '-1.5 < SPI ≤ -1', '-1 < SPI ≤ 1',
       '1 < SPI ≤ 1.5', '1.5 < SPI ≤ 2', 'SPI ≥ 2'],
             index=['Extremely dry', 'Severely dry', 'Moderately dry', 'Near normal',
       'Moderately wet', 'Severely wet', 'Extremely wet'],
                        columns=["Class"])


def reclass (spi):
    if spi <= -2:
        return "Extremely dry";
    elif -2 < spi <=-1.5:
        return "Severely dry";
    elif -1.5 < spi <=-1:
        return "Moderately dry";
    elif -1 < spi <= 1:
        return "Near normal"
    elif 1 < spi <= 1.5:
        return "Moderately wet"
    elif 1.5 < spi <= 2:
        return "Severely wet"
    elif spi >= 2:
        return "Extremely wet"
                 

class SPIgraph(tk.Frame):
    def __init__(self, parent,dataframe=None):#, controller):
        tk.Frame.__init__(self, parent)

    
        self.note = ttk.Notebook(parent)
        self.tab1 = ttk.Frame(self.note)
        self.tab2 = ttk.Frame(self.note)
        self.tab3= ttk.Frame(self.note)
        self.df=dataframe
        self.parent = parent
        self.parent.title("SPI Plot")
        self.style = ttk.Style()
        self.style.theme_use("default")
        
        self.pack(fill=tk.BOTH, expand=2)
        self.intial()
    def intial(self):
        
        
        self.note.add(self.tab1, text = "SPI graph")#,image=scheduledimage, compound=TOP)
        
        
        label = tk.Label(self.tab1, text="Graph Page!")
        label.grid(row=1)#,pady=3,padx=3)
        exit_btn=tk.Button(self.tab1,text='Go back to main page',command=self.close,
                        activebackground='grey',activeforeground='#AB78F1',
                        bg='#58F0AB',highlightcolor='red',padx='10px',pady='3px')
        exit_btn.grid(row=2,  column=2)
        num=int("%i11"%len(self.df.columns))
        i=0
        for col in self.df.columns:
            spi_pos=self.df[col].clip(lower=0).to_frame(col)            
            spi_neg=self.df[col].clip(upper=0).to_frame(col)
            i+=1

        self.frame=ttk.Frame(self.tab1)
        self.frame.grid(row=2,sticky=tk.W+tk.E)
        
        f=[]
        
        for col in self.df.columns:
            tab=ttk.Frame(self.note )
            f.append(tab)
            self.note.add(tab, text=col)
            df2=self.df[col].to_frame(col)
            df2.dropna(inplace=True)
            #print (df2.head())
            import statsmodels.api as sm
            decomposition = sm.tsa.seasonal_decompose(df2, model='additive')
            self.frame=ttk.Frame(tab)
            self.frame.grid(row=2,sticky=tk.W+tk.E)
        self.tab2=ttk.Frame(self.note )
        self.note.add(self.tab2, text="Frequency")
        num=int("%i11"%len(self.df.columns))
        i=0
        for col in self.df.columns:
            SPI=self.df[col].to_frame(col)
            print (SPI[col]) 
            SPI['spi'] = SPI[col]
            #SPI.dropna(inplace=True)
            #print('567890',SPI.head())
            #SPI["Class"]=SPI[col].apply(reclass)
            #SPIgroup=SPI.groupby(by="Class")
            #count=SPIgroup.count()
            #d=SPIclasses.join(count)
            #print(d.head())                
            i+=1
        self.frame1=ttk.Frame(self.tab2)
        self.frame1.grid(row=2,sticky=tk.W+tk.E)
        self.note.pack()
        

    def close(self):
        self.parent.destroy()
if __name__ == "__main__":
    import pandas as pd
    from fits import dateparse
    #df=pd.DataFrame(data=[2,3,4,5],columns=["x"])
    file="data/GONBADEG.csv"
    #file="data/rain_GONBADEG.csv"
    #file="data/clearn_train_spi1.csv"
    #file="data/11.csv"
    df=pd.read_csv(file,index_col=0,
                      date_parser=dateparse,parse_dates=True)
    #df=pd.read_csv(file,sep=',',
    #                  date_parser=dateparse,parse_dates=True)
    print (df)
    root = tk.Tk()
    root.resizable(width=tk.FALSE, height=tk.FALSE)
    print (root)
    app = SPIgraph(root,df)
    root.mainloop()
    
    label="SPI1"
    
    SPI=df[label]
    
    SPI=SPI.to_frame(label)
    
    SPI.dropna(inplace=True)
    SPI["Class"]=SPI[label].apply(reclass)
    SPIgroup=SPI.groupby(by="Class")
    count=SPIgroup.count()
    d=SPIclasses.join(count)

    
    
    for i in SPI.values:
        print (i)
    
        
    

    
