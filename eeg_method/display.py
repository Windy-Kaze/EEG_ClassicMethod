
def wranning(msg):
    print(f"\033[93m{msg}\033[0m")  # Yellow text
    
def plot_shark_img():

    try:
        with open( './text_toolinone.com.txt', encoding='utf-8') as file:
            content = file.read()
            print(content)  # 打印文件内容
    except Exception as e:
        print(f"{e}")
        
