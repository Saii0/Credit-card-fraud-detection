import tkinter as tk
from tkinter import ttk, LEFT, END
from tkcalendar import DateEntry
import cv2
from PIL import Image , ImageTk 
import time

import Data_processing as DPros

import sqlite3 as sql
##############################################+=============================================================

root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Credit card fraud Detection")


# img=ImageTk.PhotoImage(Image.open("bg1.jpg"))

# img2=ImageTk.PhotoImage(Image.open("bg2.jpg"))

# img3=ImageTk.PhotoImage(Image.open("bg3.jpg"))

# Load the background image
bg_image = Image.open("guimain.jpg")
bg_image = bg_image.resize((w, h), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(bg_image)

# Create a label to display the background image
background_label = tk.Label(root, image=bg_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

logo_label=tk.Label()
logo_label.place(x=0,y=0)

x = 1

# # function to change to next image
# def move():
# 	global x
# 	if x == 4:
# 		x = 1
# 	if x == 1:
# 		logo_label.config(image=img)
# 	elif x == 2:
# 		logo_label.config(image=img2)
# 	elif x == 3:
# 		logo_label.config(image=img3)
# 	x = x+1
# 	root.after(2000, move)

# # calling the function
# move()
#
label_l1 = tk.Label(root, text="Credit Card Fraud Detection",font=("Bookman Old Style", 40, 'bold')
                    , fg="black", width=25, height=1)
label_l1.place(x=270, y=20)


#_+++++++++++++++++++++++++++++++++++++++++++++++++++++++
frame_alpr = tk.LabelFrame(root, text="  Process ", width=300, height=500, font=('times', 15, ' bold '),bg="#d3d3d3")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=1100, y=150)

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def update_label(str_T):
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25),fg='black' )
    result_label.place(x=350, y=500)


def Merchant_win():
    
    from subprocess import call
    call(["python", "Merchant_Mast.py"])

def Trans_win():
    
    from subprocess import call
    call(["python", "Trans_Mast.py"])
  
class Page1():

    CCNo = ""
    Exdate = ""
    CPin = ""
    CBal = ""
    CStatus = ""
    Mode=True

    def __init__(self, *args, **kwargs):
        
#        tk.Frame.__init__(self, *args, **kwargs)
        
        cardentry = tk.Toplevel()
        cardentry.geometry("500x200+400+150")
        cardentry.title("Credit Card Master")

        cardentry.grid_rowconfigure(0, weight=1)
        cardentry.grid_rowconfigure(1, weight=0)
        cardentry.grid_columnconfigure(0, weight=1)



        ttop = tk.Frame(cardentry, width=50,bg='#d3d3d3', height=50, pady=4)
        ttop.grid(row=0, column=0, sticky=tk.NSEW)
    
        tbtn = tk.Frame(cardentry, width=50,bg='#d3d3d3', height=50, pady=4)
        tbtn.grid(row=1, column=0, sticky=tk.NSEW)

    
        CCNoLB = tk.Label(ttop, text='Credit Card No: ',bg='#d3d3d3')  # More labels
        ExdateLB = tk.Label(ttop, text='Expiry Date: ',bg='#d3d3d3')  # ^
        CPinLB = tk.Label(ttop, text='Card Pin: ',bg='#d3d3d3')  # ^
        CBalLB = tk.Label(ttop, text='Balance: ',bg='#d3d3d3')  # ^
        CStatusLB = tk.Label(ttop, text='Status: ',bg='#d3d3d3')  # ^
        CInfoLB = tk.Label(ttop, text='-------',fg='red',bg='#d3d3d3',font=('times', 15,' bold '))  # ^
        
        
        #Create string variables
        self.CCNo = tk.StringVar()
        self.Exdate = tk.StringVar()
        self.CPin = tk.StringVar()
        self.CBal = tk.DoubleVar()
        self.CStatus = tk.IntVar()
        self.Mode=tk.BooleanVar()
        self.Var=tk.IntVar()
        self.Var.set(1)

        #Set up text entry boxes
        
        CCNoEL = tk.Entry(ttop, textvariable = self.CCNo,width=40)  # The entry input
        ExdateEL= DateEntry(ttop, width=17, year=2019, month=6, day=22,background='darkblue', foreground='white', borderwidth=2, textvariable = self.Exdate)
        ExdateEL.grid(row=2, column=1,sticky=tk.W)
#        ExdateEL = tk.Entry(ttop, textvariable = self.Exdate)
        CPinEL = tk.Entry(ttop, textvariable = self.CPin)    
        CBalEL = tk.Entry(ttop, textvariable = self.CBal)

        self.Var.set(1)
        RA = tk.Radiobutton(ttop, text="Add Mode", variable=self.Var, value=1,
          bg='#d3d3d3',fg='brown')
        RA.grid( column=0, row=0,sticky=tk.W )
        
        RE = tk.Radiobutton(ttop, text="Edit Mode", variable=self.Var, value=2,
          bg='#d3d3d3',fg='brown')
        RE.grid( column=1, row=0,sticky=tk.W  )


        def sel():
            selection = str(self.CStatus.get())
            print(selection)


#        var = tk.IntVar()
        
        R1 = tk.Radiobutton(ttop, text="Active", variable=self.CStatus, value=1,
                          command=sel,bg='#d3d3d3')
        R1.grid( column=1, row=5,sticky=tk.W )
        
        R2 = tk.Radiobutton(ttop, text="Deactivate", variable=self.CStatus, value=2,
                          command=sel,bg='#d3d3d3')
        R2.grid( column=2, row=5,sticky=tk.W  )


        CCNoLB.grid(row=1, sticky=tk.W)
        ExdateLB.grid(row=2, sticky=tk.W)
        CPinLB.grid(row=3, sticky=tk.W)
        CBalLB.grid(row=4, sticky=tk.W)
        CStatusLB.grid(row=5, sticky=tk.W)
        CInfoLB.grid(row=6,  column=1,sticky=tk.EW)
    
        CCNoEL.grid(row=1, column=1,sticky=tk.W)
#        ExdateEL.grid(row=2, column=1)
        CPinEL.grid(row=3, column=1,sticky=tk.W)
        CBalEL.grid(row=4, column=1,sticky=tk.W)



        def Clear_entry():
            CCNoEL.config(state=tk.NORMAL)
            CCNoEL.delete(0, END)
            ExdateEL.delete(0, END)
            CPinEL.delete(0, END)
            CBalEL.delete(0, END)
            ExdateEL.focus_set()
       
        clrB = tk.Button(tbtn,bg='#FFD700', text='Clear',
                    command=Clear_entry,width=10)
        clrB.grid(column=0, row=4,columnspan=1)  #, sticky=tk.W)
        

        def lbl_info(msg):
           CInfoLB.config(text=msg)


#To Find a record
        
        def FindDetails(cNumber):
        
            conn = sql.connect('C:/Users/Atharv/Downloads/Credit_Card Adaboost Algo/Credit_Card Adaboost Algo/creditcarddb.db')
            c = conn.cursor()
            c.execute('PRAGMA foreign_keys = ON')
            conn.commit()
    
            """
            Returns a entry by the given id
            """
            c.execute(f"SELECT * FROM card_master WHERE cardno ='{cNumber}'")
            
            return c.fetchone()
    
    
            c.close()
            conn.close()
            
#For delete the record
        
        def Deletedetails():
            Del_rec=FindDetails(self.CCNo.get())
            
            if Del_rec is not None:
                conn = sql.connect('creditcarddb.db')
                c = conn.cursor()
                c.execute('PRAGMA foreign_keys = ON')
                conn.commit()
        
                """
                Delete a entry by the given id
                """
                delsql="Delete from card_master WHERE cardno='{0}'".format(self.CCNo.get())
                lbl_info("------Record Deleted------")
                c.execute(delsql)
                conn.commit()
                
                
                Clear_entry()
                c.close()
                conn.close()
            else:
                lbl_info("------Record Not Deleted------")
                
#For Saveing the detail
                            
        def SaveDetails():
            
            conn = sql.connect('C:/Users/Atharv/Downloads/Credit_Card Adaboost Algo/Credit_Card Adaboost Algo/creditcarddb.db')
            c = conn.cursor()
            c.execute('PRAGMA foreign_keys = ON')
            conn.commit()
    
            """
            Creates the Table if it does not exist
            """
            
            sqlstr= "CREATE TABLE IF NOT EXISTS card_master(id INTEGER PRIMARY KEY AUTOINCREMENT,cardno TEXT,expiredate TEXT, pin TEXT, balance FLOAT,status INTEGER)"
    
            c.execute(sqlstr)
    
            customerData = [(None, self.CCNo.get(), self.Exdate.get(), self.CPin.get(), self.CBal.get(), self.CStatus.get())]
    
            if self.Mode:
                for element in customerData:
                    
                    c.execute("INSERT INTO card_master VALUES (?,?,?,?,?,?)", element)
                conn.commit()
                lbl_info("--------Saved New Record--------")
                Clear_entry()
            else:
    
                c.execute("""UPDATE card_master SET expiredate='{0}', pin='{1}', balance={2}, 
                          status={3} WHERE cardno='{4}' """.format(self.Exdate.get(), 
                          self.CPin.get(), self.CBal.get(),self.CStatus.get(),self.CCNo.get()))
                conn.commit()
                
                lbl_info("--------Saved Updated Record--------")
                CCNoEL.config(state=tk.NORMAL)
                
            c.close()
            conn.close()
  

        #Save Details button
        saveCustomerDetails = tk.Button(tbtn,bg='#FFD700', width=10,text = "Save Details", command = SaveDetails)
        saveCustomerDetails.grid(row=4,column=1,columnspan=1, padx=5)


        DelB = tk.Button(tbtn, text='Delete Entry',bg='#FFD700', fg='black',
                    command=Deletedetails,width=10) 
        DelB.grid(column=2, row=4,columnspan=1, padx=5)

#        ExtB = tk.Button(tbtn, text='Exit',bg='cyan3', fg='red',
#                    command=window,width=10) 
#        ExtB.grid(column=3, row=4,columnspan=1, padx=5)
        
        CCNoEL.focus()
        
           
    
        def on_focus_out(event):
            
            cardDetail=()
            
            if CCNoEL.get()!="":
                
                cardDetail=FindDetails(CCNoEL.get())
#                Clear_entry()
                if cardDetail is not None:
                    
                    CInfoLB.config(text="---------Edit Mode----------")
                    self.Mode=False
                    
                    CCNoEL.config(state=tk.DISABLED)
                    CPinEL.insert(0,cardDetail[3])
                    CBalEL.insert(0,cardDetail[4])
                    ExdateEL.insert(0,cardDetail[2])
    
                    if int(cardDetail[5])==1:
                        R1.select()
                    else:
                        R2.select()    
                else:
                    CInfoLB.config(text="---------ADD Mode---------") 
                    self.Mode=True
                    
            
            ExdateEL.focus_set()


        CCNoEL.bind('<FocusOut>', on_focus_out)
        
        
        cardentry.mainloop()
            
        
##################################################################################################################
def card_entry():
    card_frm=Page1()
    

#################################################################################################################
def window():
    root.destroy()

# def CL_SVM():
#     update_label("Process Start...............")
#     start = time.time()

#     X=DPros.SVM_Cl()
    
#     end = time.time()
        
#     ET="Execution Time: {0:.5} seconds \n".format(end-start)
#     msg=X+'\n'+ET

#     update_label(msg)




def CL_AdaBoost():
    
    update_label("Process Start...............")
    
    start = time.time()

    X=DPros.AdaBoost_Cl()
    
    end = time.time()
        
    ET="Execution Time: {0:.5} seconds \n".format(end-start)
    msg=X+'\n'+ET

    update_label(msg)
    

button1 = tk.Button(frame_alpr, text="Card Master", command=card_entry ,width=15, height=1, font=('times', 15, ' bold '),bg="#d3d3d3",fg="black")
button1.place(x=50, y=30)

button2 = tk.Button(frame_alpr, text="Merchant Master", command=Merchant_win, width=15, height=1, font=('times', 15, ' bold '),bg="#d3d3d3",fg="black")
button2.place(x=50, y=90)

button3 = tk.Button(frame_alpr, text="Transaction", command=Trans_win, width=15, height=1, font=('times', 15, ' bold '),bg="#d3d3d3",fg="black")
button3.place(x=50, y=150)

# button4 = tk.Button(frame_alpr, text="SVM", command=CL_SVM,width=15, height=1,bg="Darkgoldenrod2",fg="blue", font=('times', 15, ' bold '))
# button4.place(x=50, y=210)

button4 = tk.Button(frame_alpr, text="Adaboost", command=CL_AdaBoost,width=15, height=1,bg="#d3d3d3",fg="black", font=('times', 15, ' bold '))
button4.place(x=50, y=210)


exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="red",fg="black")
exit.place(x=50, y=270)



root.mainloop()
