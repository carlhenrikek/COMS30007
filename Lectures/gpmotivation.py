import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
#import seaborn as sns
#import GPy


def gp_prediction(x1,y1,lengthScale,varSigma,noise):

    x = np.linspace(-6,6,200)
    x = x.reshape(200,1)
    
    d = x-x1
    k_starX = varSigma*np.exp(-np.power(d,2)/lengthScale) + noise
    k_xx = varSigma + noise;
    k_starstar = varSigma*np.exp(-np.power(cdist(x,x),2)/lengthScale)+noise
    
    mu = k_starX.T/k_xx*y1
    var = k_starstar - k_starX.T/k_xx*k_starX
    
    return mu, var, x

def conditional_normal(mu,sigma,x1):
    var = sigma[1,1]-sigma[1,0]*(1.0/sigma[0,0])*sigma[0,1]
    mean = sigma[1,0]*(1.0/sigma[0,0])*x1

    pdf = multivariate_normal(mean,var)

    return pdf

def rbf_kernel_compute(x1,x2,lengthScale,varSigma,noise):

    d = x1-x2
    K = np.zeros((2,2))
    K[0,0] = varSigma
    K[1,1] = varSigma
    covar = varSigma*np.exp(-np.power(d,2)/lengthScale)
    K[0,1] = covar
    K[1,0] = covar

    K = K + noise
    
    # add diagonal term for numerical calc
    K = K + 0.001*np.diag(np.random.rand(2))
    
    return K

def plot_pdf(ax,x1,x2,y1,lengthScale,varSigma,noise):
    ax.clear()

    K = rbf_kernel_compute(x1,x2,lengthScale,varSigma,noise)
    
    # plot joint
    N = 100
    mu = np.array([0.0,0.0])
    pdf = multivariate_normal(mu,K)
    x = np.linspace(-4,4,N)
    y = np.linspace(-4,4,N)
    x1p,x2p = np.meshgrid(x,y)
    pos = np.vstack((x1p.flatten(),x2p.flatten()))
    pos = pos.T
    Z = pdf.pdf(pos)
    Z = Z.reshape(N,N)

    levels = np.linspace(0.01,0.3,5)
    
    pdf_c = ax.contour(x1p,x2p,Z,levels,colors='k')

    # plot conditional 
    cpdf = conditional_normal(0.0,K,y1)
    y = np.linspace(-3,3,100)
    Z = cpdf.pdf(y)
    
    ax.plot(Z+y1,y,lw=3,color='r')

    cpdf = multivariate_normal(y1,0.001)
    
    x = np.linspace(-3,3,100)
    Z = cpdf.pdf(x)

    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    
    #ax.plot(x,Z+y1,'r')
    
    return pdf_c
    
def plot_lines(ax,x1,x2,y1,col1,col2,lengthScale,varSigma,noise):

    ax.clear()

    if not draw_process:
        y = [-3,3]
        x = [x1,x1]
        ax.plot(x,y,col1,lw=3)
        x = [x2,x2]
        ax.plot(x,y,col2,lw=3)
        x = [-4,4]
        ax.plot(x,[y1,y1],'k',lw=1.2,ls='--')

    ax.scatter(x1,y1,200,'k','x')

    # plot conditional distribution
    K = rbf_kernel_compute(x1,x2,lengthScale,varSigma,noise)
    cpdf = conditional_normal(0.0,K,y1)
    y = np.linspace(-3,3,100)
    Z = cpdf.pdf(y)

    ax.plot(Z+x2,y,'r',lw=3)
    #ax.plot([-4,4],[cpdf.mean,cpdf.mean],'b',lw=0.3)

    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    # plot full GP
    if draw_process:        
        mu, var, x = gp_prediction(x1,y1,lengthScale,varSigma,noise)
        
        mu = mu.flatten()
        var = np.diag(var)
        var = var.flatten()
        x = x.flatten()
        
        ax.fill_between(x,-varSigma*np.ones(len(x)),varSigma*np.ones(len(x)),alpha=0.3,facecolor='blue')
        ax.fill_between(x,mu-var,mu+var,alpha=0.3,facecolor='green')
        ax.plot(x,mu,'k',ls='dashed',lw=3)

    ax.set_xlim([-4,4])
    ax.set_ylim([-3,3])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return
        
fig = plt.figure()
ax_pdf = fig.add_subplot(121)
ax_pdf.set_autoscaley_on(False)
ax_pdf.set_autoscalex_on(False)
ax_pdf.axis([-4,-4,4,4])
ax_curve = fig.add_subplot(122)
ax_curve.set_autoscaley_on(False)
ax_curve.set_autoscalex_on(False)
ax_curve.axis([-4,-2,4,2])
plt.subplots_adjust(left=0.1,bottom=0.3)

ax_curve.axhline(y=0, color='k')
ax_curve.axvline(x=0, color='k')

axcolor = 'lightgoldenrodyellow'

# initial value
x1 = -3.04
x2 = 1.67
y1 = -1.0
lengthScale = 155.0
varSigma = 0.76
noise = 0.000001
draw_process = False

# first plots
plot_data_pdf = plot_pdf(ax_pdf,x1,x2,y1,lengthScale,varSigma,noise)
plot_data_lines_1 = plot_lines(ax_curve,x1,x2,y1,'g','r',lengthScale,varSigma,noise)

ax_x1 = plt.axes([0.25, 0.20, 0.65, 0.03])#, axisbg=axcolor)
ax_y1 = plt.axes([0.25, 0.15, 0.65, 0.03])#, axisbg=axcolor)
ax_x2 = plt.axes([0.25, 0.175, 0.65, 0.03])#, axisbg=axcolor)
ax_lengthScale = plt.axes([0.25,0.125, 0.65, 0.03])#,axisbg=axcolor)
ax_varSigma = plt.axes([0.25,0.1, 0.65, 0.03])#,axisbg=axcolor)
ax_noise = plt.axes([0.25,0.075, 0.65, 0.03])#,axisbg=axcolor)

slider_x1 = Slider(ax_x1,'X1',-4.0,4.0,valinit=x1)
slider_x2 = Slider(ax_x2,'X2',-4.0,4.0,valinit=x2)
slider_y1 = Slider(ax_y1,'Y1',-2.0,2.0,valinit=y1)
slider_lengthScale = Slider(ax_lengthScale,'LengthScale',0.0,400.0,valinit=lengthScale)
slider_varSigma = Slider(ax_varSigma,'Variance',0.0,4.0,valinit=varSigma)
slider_noise = Slider(ax_noise,'Noise',0.0,4.0,valinit=noise)

#def onclick(event):
#    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#          (event.button, event.x, event.y, event.xdata, event.ydata))
#cid = fig.canvas.mpl_connect('button_press_event', onclick)


def update(val):
    global x1,x2,y1,lengthScale, varSigma, noise
    
    x1 = slider_x1.val
    x2 = slider_x2.val
    y1 = slider_y1.val
    lengthScale = slider_lengthScale.val
    varSigma = slider_varSigma.val
    noise = slider_noise.val
    
    # update plots
    plot_lines(ax_curve,x1,x2,y1,'g','r',lengthScale,varSigma,noise)
    plot_pdf(ax_pdf,x1,x2,y1,lengthScale,varSigma,noise)
slider_x1.on_changed(update)
slider_x2.on_changed(update)
slider_y1.on_changed(update)
slider_lengthScale.on_changed(update)
slider_varSigma.on_changed(update)
slider_noise.on_changed(update)

# reset button
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    slider_x1.reset()
    slider_x2.reset()
    slider_y1.reset()
    slider_lengthScale.reset()
    slider_varSigma.reset()
    slider_noise.reset()
button.on_clicked(reset)
button.on_clicked(update)

# sample button
sampleax = plt.axes([0.025, 0.0, 0.1, 0.1])#, axisbg=axcolor)
button_sample = Button(sampleax, 'Sample' ,color=axcolor, hovercolor='0.975')
def sample_gp(event):
    mu, var, x = gp_prediction(x1,y1,lengthScale,varSigma,noise)
    f_star = np.random.multivariate_normal(mu.flatten(),var,20)
    for i in range(0,20):
        ax_curve.plot(x,f_star[i])
    
button_sample.on_clicked(sample_gp)


# buttons
rax = plt.axes([0.025, 0.1, 0.1, 0.1])#, axisbg=axcolor)
radio = RadioButtons(rax, ('Distr', 'Process'), active=0)

def buttonfunc(label):
    global draw_process
    if label=='Process':
        draw_process = True;
    else:
        draw_process = False;
radio.on_clicked(buttonfunc)
radio.on_clicked(update)

plt.show()
