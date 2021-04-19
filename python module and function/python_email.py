

#%%
def send_email_attach(from_who,to_who,subject_name,content,attach_file_name,file_path,password):

    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email import encoders 
   
    fromaddr = from_who
    toaddr = to_who
   
# instance of MIMEMultipart 
    msg = MIMEMultipart() 
  
# storing the senders email address   
    msg['From'] = fromaddr 
  
# storing the receivers email address  
    msg['To'] = toaddr 
  
# storing the subject  
    msg['Subject'] = subject_name
  
# string to store the body of the mail 
    body = content
  
# attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 
  
# open the file to be sent  
    filename = attach_file_name
    attachment = open(file_path, "rb") 
  
# instance of MIMEBase and named as p 
    p = MIMEBase('application', 'octet-stream') 
  
# To change the payload into encoded form 
    p.set_payload((attachment).read()) 
  
# encode into base64 
    encoders.encode_base64(p) 
   
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
  
# attach the instance 'p' to instance 'msg' 
    msg.attach(p) 
  
# creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
    s.starttls() 
  
# Authentication 
    s.login(fromaddr, password) 
  
# Converts the Multipart msg into a string 
    text = msg.as_string() 
  
# sending the mail 
    s.sendmail(fromaddr, toaddr, text) 
    attachment.close() 
# terminating the session 
    s.quit() 

#%%
# Importing libraries 

ORG_EMAIL   = "@gmail.com"
FROM_EMAIL  = "zixiangjin921" + ORG_EMAIL
FROM_PWD    = "Intelcorei7"
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT   = 993
mail = imaplib.IMAP4_SSL(SMTP_SERVER)
mail.login(FROM_EMAIL,FROM_PWD)
mail.select('inbox')

type, data = mail.search(None, 'ALL')
mail_ids = data[0]
id_list = mail_ids.split()

first_email_id = int(id_list[0])
latest_email_id = int(id_list[-1])




#%%
'''get top n email tempelate'''

import imaplib
import email
from email.header import decode_header
import webbrowser
import os

# account credentials
username = "zixiangjin921@gmail.com"
password = "Intelcorei7"

# number of top emails to fetch
N = 3

# create an IMAP4 class with SSL, use your email provider's IMAP server
imap = imaplib.IMAP4_SSL("imap.gmail.com")
# authenticate
imap.login(username, password)

# select a mailbox (in this case, the inbox mailbox)
# use imap.list() to get the list of mailboxes
status, messages = imap.select("INBOX")

# total number of emails
messages = int(messages[0])

for i in range(messages-4, messages-N-4, -1):
    # fetch the email message by ID
    res, msg = imap.fetch(str(i), "(RFC822)")
    for response in msg:
        if isinstance(response, tuple):
            # parse a bytes email into a message object
            msg = email.message_from_bytes(response[1])
            # decode the email subject
            subject = decode_header(msg["Subject"])[0][0]
            if isinstance(subject, bytes):
                # if it's a bytes, decode to str
                subject = subject.decode()
            # email sender
            from_ = msg.get("From")
            print("Subject:", subject)
            print("From:", from_)
            # if the email message is multipart
            if msg.is_multipart():
                # iterate over email parts
                for part in msg.walk():
                    # extract content type of email
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    try:
                        # get the email body
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        # print text/plain emails and skip attachments
                        print(body)
                    elif "attachment" in content_disposition:
                        # download attachment
                        filename = part.get_filename()
                        if filename:
                            if not os.path.isdir(subject):
                                # make a folder for this email (named after the subject)
                                os.mkdir(subject)
                            filepath = os.path.join(subject, filename)
                            # download attachment and save it
                            open(filepath, "wb").write(part.get_payload(decode=True))
            else:
                # extract content type of email
                content_type = msg.get_content_type()
                # get the email body
                body = msg.get_payload(decode=True).decode()
                if content_type == "text/plain":
                    # print only text email parts
                    print(body)
            if content_type == "text/html":
                # if it's HTML, create a new HTML file and open it in browser
                if not os.path.isdir(subject):
                    # make a folder for this email (named after the subject)
                    os.mkdir(subject)
                filename = f"{subject[:50]}.html"
                filepath = os.path.join(subject, filename)
                # write the file
                open(filepath, "w").write(body)
                # open in the default browser
                webbrowser.open(filepath)

            print("="*100)

# close the connection and logout
imap.close()
imap.logout()



#%%

 
def send_email_only(from_addr, to_addr_list, cc_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    import smtplib
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message
 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems

# %%   
sendemail(from_addr    = 'zixiangjin921@gmail.com', 
          to_addr_list = ['zjin24@asu.edu'],
          cc_addr_list = ['zixiangjin921@163.com'], 
          subject      = 'python email', 
          message      = 'babe hope you can see this email which send from python', 
          login        = 'zixiangjin921@gmail.com', 
          password     = 'Intelcorei7')


# %%
email_list=['andrew.weflen@maricopa.gov','zjin24@asu.edu','richard.langevin@maricopa.gov','gerry.gill@maricopa.gov','Matthew.Melendez@maricopa.gov','Ashley.Miranda@maricopa.gov']
for i in email_list:

    send_email_attach(
        from_who='zixiangjin921@gmail.com',
    to_who=i,subject_name='Beltmann Dailt inventory update',
    content='Inventory data exclude Vault 12 week. data from smartsheet, update date 7/16/2020 9:21 am',
    attach_file_name='beltmann inventory.png',
    file_path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/image/beltmann inventory 20200716 091907832611.png',
    password='Intelcorei7')
#send_email_attach('zixiangjin921@gmail.com','jyuan42@asu.edu','Send from Python','真是非常的厉害','photo grad.png','C:/Users/Mr.Goldss/Desktop/qw.png','Intelcorei7')

# %%


import imaplib, email 
  
user = 'zixiangjin921@gmail.com'
password = 'Intelcorei7'
imap_url = 'imap.gmail.com'
  
# Function to get email content part i.e its body part 
def get_body(msg): 
    if msg.is_multipart(): 
        return get_body(msg.get_payload(0)) 
    else: 
        return msg.get_payload(None, True) 
  
# Function to search for a key value pair  
def search(key, value, con):  
    result, data = con.search(None, key, '"{}"'.format(value)) 
    return data 
  
# Function to get the list of emails under this label 
def get_emails(result_bytes): 
    msgs = [] # all the email data are pushed inside an array 
    for num in result_bytes[0].split(): 
        typ, data = con.fetch(num, '(RFC822)') 
        msgs.append(data) 
  
    return msgs 
  
# this is done to make SSL connnection with GMAIL 
con = imaplib.IMAP4_SSL(imap_url)  
  
# logging the user in 
con.login(user, password)  
  
# calling function to check for email under this label 
con.select('Inbox')  
  
 # fetching emails from this user "tu**h*****1@gmail.com" 
msgs = get_emails(search('FROM', 'support@kite.com', con)) 
  
# Uncomment this to see what actually comes as data  
# print(msgs)  
  
  
# Finding the required content from our msgs 
# User can make custom changes in this part to 
# fetch the required content he / she needs 
  
# printing them by the order they are displayed in your gmail  
for msg in msgs[::-1]:  
    for sent in msg: 
        if type(sent) is tuple:  
  
            # encoding set as utf-8 
            content = str(sent[1], 'utf-8')  
            data = str(content) 
  
            # Handling errors related to unicodenecode 
            try:  
                indexstart = data.find("ltr") 
                data2 = data[indexstart + 5: len(data)] 
                indexend = data2.find("</div>") 
  
                # printtng the required content which we need 
                # to extract from our email i.e our body 
                print(data2[0: indexend]) 
  
            except UnicodeEncodeError as e: 
                pass

# %%
