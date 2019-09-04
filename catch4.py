import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
# import psutil as p
import modbus_tk.modbus_tcp as mt
import modbus_tk.defines as md
from matplotlib.widgets import Button
import threading
import time
import pandas as pd
FILE_PATH = "d:\\file.csv"
POINTS = 300
# 暂停的时候应该把本地数据读出来，或者全屏的时候把本地数据读出来

fig, ax = plt.subplots()
ax.set_ylim([0, 20000])
ax.set_xlim([0, POINTS])
ax.set_autoscale_on(False)
# ax.set_xticks([])
# ax.set_yticks(range(0, 20000, 1000))
ax.grid(True)
user = [None] * POINTS
user2 = [None]*POINTS
user3 = [None]*POINTS
user4 = [None]*POINTS


l_user, = ax.plot(range(POINTS), user, label='AI2 %')
l_user2, = ax.plot(range(POINTS), user2, label='AI9 %')
l_user3, = ax.plot(range(POINTS), user3, label='AI10 %')
l_user4, = ax.plot(range(POINTS), user4, label='AI14 %')


ax.legend(loc='upper center', ncol=4,
          prop=font_manager.FontProperties(size=10))
bg = fig.canvas.copy_from_bbox(ax.bbox)
timer = fig.canvas.new_timer(interval=100)


class Index(object):
    flag = 0

    def pause(self, event):
        global timer, user
        text = bpause.label.get_text()
        if text == "stop":
            timer.stop()
            bpause.label.set_text("start")
            ax.set_xlim([len(user)-POINTS, len(user)])

            l_user.set_data(range(len(user)), user)
            l_user2.set_data(range(len(user2)), user2)
            l_user3.set_data(range(len(user3)), user3)
            l_user4.set_data(range(len(user4)), user4)
            while True:
                try:
                    ax.draw_artist(l_user)
                    ax.draw_artist(l_user2)
                    ax.draw_artist(l_user3)
                    ax.draw_artist(l_user4)
                    break
                except:
                    pass
            ax.figure.canvas.draw()
        else:
            timer.start()
            bpause.label.set_text("stop")
            l_user.set_xdata(range(POINTS))
            l_user2.set_xdata(range(POINTS))
            l_user3.set_xdata(range(POINTS))
            l_user4.set_xdata(range(POINTS))
            ax.set_xlim([0, POINTS])

    def full(self, event):
        # if len(user) > 300:
        #     ax.set_xlim([300, len(user)])
        #     l_user.set_data(range(300, len(user)), user[300:])
        #     l_user2.set_data(range(300, len(user2)), user2[300:])
        #     l_user3.set_data(range(300, len(user3)), user3[300:])
        #     l_user4.set_data(range(300, len(user4)), user4[300:])
        text = bpause.label.get_text()
        if text == 'stop':
            self.pause(0)
        df1 = pd.read_csv(FILE_PATH)
        if len(df1.values) > POINTS:
            print("in len > 0")
            ax.set_xlim([POINTS, len(df1.iloc[:, 0])])
            l_user.set_data(
                range(POINTS, len(df1.iloc[:, 0])), df1.iloc[POINTS:, 0])
            l_user2.set_data(
                range(POINTS, len(df1.iloc[:, 1])), df1.iloc[POINTS:, 1])
            l_user3.set_data(
                range(POINTS, len(df1.iloc[:, 2])), df1.iloc[POINTS:, 2])
            l_user4.set_data(
                range(POINTS, len(df1.iloc[:, 3])), df1.iloc[POINTS:, 3])
            while True:
                try:
                    ax.draw_artist(l_user)
                    ax.draw_artist(l_user2)
                    ax.draw_artist(l_user3)
                    ax.draw_artist(l_user4)
                    break
                except:
                    pass
            ax.figure.canvas.draw()
        del df1


callback = Index()

axpause = plt.axes([0.92, 0.01, 0.075, 0.075])
bpause = Button(axpause, 'stop')
bpause.on_clicked(callback.pause)

axfull = plt.axes([0.92, 0.1, 0.075, 0.075])
bfull = Button(axfull, 'full')
bfull.on_clicked(callback.full)


y_count = 0


def get_iw(address, lower, upper):

    global timer
    global callback
    global y_count
    y_count += 1
    if y_count > 80:
        y_count = 0
        tmp = (5000,)
    elif y_count > 40:
        tmp = (15000,)
    else:
        tmp = (5000,)
    # tmp = master.execute(slave=1,function_code=md.READ_INPUT_REGISTERS,starting_address=address,quantity_of_x=1)
    return tmp, tmp in range(lower, upper + 1)


sem_save = threading.Semaphore(value=0)


def save_csv():
    global user, user2, user3, user4
    while True:
        sem_save.acquire()
        if RUN_FLAG == 0:
            break
        print("save thread is doing")
        df = pd.DataFrame()
        df['ai2'] = user[:-POINTS]
        df['ai9'] = user2[:-POINTS]
        df['ai10'] = user3[:-POINTS]
        df['ai14'] = user4[:-POINTS]
        df.to_csv(FILE_PATH, mode='a', index=False, header=False)
        del df
        user = user[-POINTS:]
        user2 = user2[-POINTS:]
        user3 = user3[-POINTS:]
        user4 = user4[-POINTS:]


TIME_COUNT_FLAG = 0


def OnTimer(ax):
    global TIME_COUNT_FLAG
    TIME_COUNT_FLAG += 1
    if TIME_COUNT_FLAG > 3000:  # 5 min
        TIME_COUNT_FLAG = 0
        sem_save.release()

    l_user.set_ydata(user[-POINTS:])
    l_user2.set_ydata(user2[-POINTS:])
    l_user3.set_ydata(user3[-POINTS:])
    l_user4.set_ydata(user4[-POINTS:])
    while True:
        try:
            ax.draw_artist(l_user)
            ax.draw_artist(l_user2)
            ax.draw_artist(l_user3)
            ax.draw_artist(l_user4)
            break
        except:
            pass
    ax.figure.canvas.draw()


def start_monitor():
    global timer
    timer.add_callback(OnTimer, ax)
    timer.start()
    plt.show()


def modbus_thread():
    global RUN_FLAG
    while True:
         # value = master.execute(slave=1,function_code=md.READ_HOLDING_REGISTERS,starting_address=1,quantity_of_x=1)
         # value = master.execute(slave=1,function_code=md.READ_INPUT_REGISTERS,starting_address=1,quantity_of_x=1)
        tmp, b = get_iw(1, 4332, 13343)
        tmp2, b1 = get_iw(8, 4302, 13292)
        tmp3, b2 = get_iw(9, 4311, 13307)
        tmp4, b3 = get_iw(13, 4315, 13310)
        user.append(tmp[0])
        user2.append(tmp2[0])
        user3.append(tmp3[0])
        user4.append(tmp4[0])

        if b or b1 or b2 or b3:
            pass
            # callback.pause(0)
        if RUN_FLAG == 0:
            break
        time.sleep(0.1)


if __name__ == '__main__':
    RUN_FLAG = 1
    df = pd.DataFrame()
    df.to_csv(FILE_PATH, index=False, header=False)
    del df
    # master = mt.TcpMaster("192.168.1.211",502)
    # # master = mt.TcpMaster("192.168.1.66",502)
    # master.set_timeout(3.0)
    mod_t = threading.Thread(target=modbus_thread, name='modbus tcp')
    mod_t.start()
    save_t = threading.Thread(target=save_csv, name='save data')
    save_t.start()
    start_monitor()
    RUN_FLAG = 0

    df = pd.DataFrame()
    df['ai2'] = user
    df['ai9'] = user2
    df['ai10'] = user3
    df['ai14'] = user4
    df.to_csv(FILE_PATH, mode='a', index=False, header=False)
    del df

    sem_save.release()
