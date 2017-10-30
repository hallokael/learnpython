from tkinter import *
import tkinter.messagebox as messagebox

class Application( Frame ):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.pack()
        self.createWidgets()
    def createWidgets(self):
        self.helloLabel=Label( self,text='Hello,ASD!' )
        self.helloLabel.pack()
        self.inp=Entry( self )
        self.inp.pack()
        self.quitButton=Button( self,text='input',command=self.WWW )
        self.quitButton.pack()
    def WWW(self  ):
        messagebox.showinfo('MESSAGE',self.inp.get())
app=Application()
app.master.title( 'hello!!' )
app.mainloop()