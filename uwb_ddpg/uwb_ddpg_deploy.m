clear; clc; close all;

%% PARALLEL POOL
nCores=feature('numcores');
cl=parcluster('local'); maxW=cl.NumWorkers;
nUse=min(maxW,max(1,floor(nCores*0.8)));
fprintf('Cores: %d | Workers: %d\n',nCores,nUse);
if isempty(gcp('nocreate')), parpool('local',nUse); end

if ~exist('ddpg_trained.mat','file')
    error('Run uwb_ddpg_train.m first.');
end
load('ddpg_trained.mat');
fprintf('Agent loaded.\n');

T=60; t=0:dt:T; N=length(t);

%% BOUSTROPHEDON PATH
lanes=8; ly=linspace(1,H-1,lanes); px=[]; py=[];
for L=1:lanes
    if mod(L,2)==1,px=[px 1 W-1];else,px=[px W-1 1];end
    py=[py ly(L) ly(L)];
end
dc=[0 cumsum(sqrt(diff(px).^2+diff(py).^2))];
dq=linspace(0,dc(end),N);
true_x=interp1(dc,px,dq); true_y=interp1(dc,py,dq);
true_range=sqrt(true_x.^2+true_y.^2);

rng(7);
raw=max(true_range+0.25*randn(1,N)+(rand(1,N)<0.12).*(0.4+0.8*rand(1,N)),0);
nlos_mask=(raw-true_range)>0.25;

%% FIXED KALMAN
Q_fix=diag([0.0005 0.005]); R_fix=0.25^2;
x_fix=[raw(1);0]; P_fix=eye(2); kf_fix=zeros(1,N);
for k=1:N
    x_fix=F_k*x_fix; P_fix=F_k*P_fix*F_k'+Q_fix;
    Kf=P_fix*H_k'/(H_k*P_fix*H_k'+R_fix);
    x_fix=x_fix+Kf*(raw(k)-H_k*x_fix);
    P_fix=(eye(2)-Kf*H_k)*P_fix;
    kf_fix(k)=max(0,x_fix(1));
end

%% HELPER FUNCTIONS
function a=act_fwd(n,s)
    a=tanh(n.W3*max(0,n.W2*max(0,n.W1*s+n.b1)+n.b2)+n.b3);
end
function [qp,qv,rv]=decode(a,qpn,qpx,qvn,qvx,rn,rx)
    qp=max(qpn,min(qpx, qpn+(a(1)+1)/2*(qpx-qpn)));
    qv=max(qvn,min(qvx, qvn+(a(2)+1)/2*(qvx-qvn)));
    rv=max(rn, min(rx,  rn +(a(3)+1)/2*(rx-rn)));
end
function s=norm_s(r3,e3,vel,nlos,W,H)
    s=[r3/sqrt(W^2+H^2); e3/100; vel/10; double(nlos)];
end

%% PRE-COMPUTE DDPG
fprintf('Pre-computing DDPG outputs...\n');
D_MAX=sqrt(W^2+H^2);
kf_rl=zeros(1,N); qp_hist=zeros(1,N); rv_hist=zeros(1,N);
x_rl=[raw(1);0]; P_rl=eye(2);
raw_hist=raw(1)*ones(3,1); err_hist=zeros(3,1);
state=norm_s(raw_hist,err_hist,0,0,W,H);

for k=1:N
    a_out=act_fwd(actor,state);
    a_out(isnan(a_out)|isinf(a_out))=0;
    [qp,qv,rv]=decode(a_out,Q_pos_min,Q_pos_max,Q_vel_min,Q_vel_max,R_min,R_max);

    Q_rl=diag([qp qv]); R_rl=rv^2;
    x_rl=F_k*x_rl; P_rl=F_k*P_rl*F_k'+Q_rl;
    K_rl=P_rl*H_k'/(H_k*P_rl*H_k'+R_rl);
    x_rl=x_rl+K_rl*(raw(k)-H_k*x_rl);
    P_rl=(eye(2)-K_rl*H_k)*P_rl;

    kf_rl(k)=max(0,min(D_MAX,x_rl(1)));
    if isnan(kf_rl(k))|isinf(kf_rl(k))
        x_rl=[raw(k);0]; P_rl=eye(2)*0.5; kf_rl(k)=raw(k);
    end
    qp_hist(k)=qp; rv_hist(k)=rv;

    vel_est=0; if k>1, vel_est=(raw(k)-raw(k-1))*fs; end
    err_now=(kf_rl(k)-true_range(k))*100;
    raw_hist=[raw_hist(2:end);raw(k)];
    err_hist=[err_hist(2:end);err_now];
    state=norm_s(raw_hist,err_hist,vel_est,double(nlos_mask(k)),W,H);
end

%% ERRORS & RUNNING RMSE
err_raw=(raw    -true_range)*100;
err_fix=(kf_fix -true_range)*100;
err_rl =(kf_rl  -true_range)*100;
rmse_fix=zeros(1,N); rmse_rl=zeros(1,N);
af=0; ar=0;
for k=1:N
    af=af+err_fix(k)^2; rmse_fix(k)=sqrt(af/k);
    ar=ar+err_rl(k)^2;  rmse_rl(k) =sqrt(ar/k);
end
fprintf('Pre-compute done.\n\n');
fprintf('Fixed KF RMSE : %.4f cm\n',rmse_fix(N));
fprintf('DDPG  KF RMSE : %.4f cm\n',rmse_rl(N));
fprintf('Improvement   : %.2f%%\n\n',(1-rmse_rl(N)/rmse_fix(N))*100);

%% COLOURS & STYLE
Ctr=[0.40 0.40 0.40]; Craw=[0.92 0.22 0.08];
Cfix=[0.50 0.10 0.80]; Crl=[0.05 0.72 0.32];
Cq=[0.92 0.55 0.05];   Cr=[0.08 0.45 0.88];
LW=0.7; FS=9; GS=[0.985 0.985 0.985];

%% FIVE SEPARATE FIGURE WINDOWS
f1=figure('Color','w','Position',[10  410 720 370],'Name','Range','NumberTitle','off');
f2=figure('Color','w','Position',[740 410 720 370],'Name','Q & R Adaptation','NumberTitle','off');
f3=figure('Color','w','Position',[10  20  720 370],'Name','Ranging Error','NumberTitle','off');
f4=figure('Color','w','Position',[740 20  720 370],'Name','Running RMSE','NumberTitle','off');
f5=figure('Color','w','Position',[1470 20 430 760],'Name','Trajectory','NumberTitle','off');

%% AX1 — RANGE
ax1=axes(f1,'Position',[0.09 0.13 0.88 0.82]); hold on;
h_tr =animatedline(ax1,'Color',Ctr, 'LineWidth',LW,'DisplayName','True');
h_raw=animatedline(ax1,'Color',Craw,'LineWidth',LW,'DisplayName','Raw');
h_fix=animatedline(ax1,'Color',Cfix,'LineWidth',LW,'DisplayName','Fixed KF');
h_rl =animatedline(ax1,'Color',Crl, 'LineWidth',LW,'DisplayName','DDPG KF');
ylim(ax1,[0 22]); xlim(ax1,[0 10]);
set(ax1,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',FS,...
    'YTick',0:1:22,'XTick',0:0.5:10,'TickDir','in');
grid(ax1,'on');
xlabel(ax1,'Time (s)','FontSize',FS); ylabel(ax1,'Range (m)','FontSize',FS);
legend(ax1,'True','Raw','Fixed KF','DDPG KF','Location','best','FontSize',7,'Box','off');
title(ax1,'Distance from Anchor','FontSize',10,'FontWeight','bold');

%% AX2 — Q & R
ax2=axes(f2,'Position',[0.09 0.13 0.88 0.82]); hold on;
h_qp=animatedline(ax2,'Color',Cq,'LineWidth',LW,'DisplayName','Q_{pos}');
h_rv=animatedline(ax2,'Color',Cr,'LineWidth',LW,'DisplayName','R');
ylim(ax2,[0 0.55]); xlim(ax2,[0 10]);
set(ax2,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',FS,...
    'YTick',0:0.025:0.55,'XTick',0:0.5:10,'TickDir','in');
grid(ax2,'on');
xlabel(ax2,'Time (s)','FontSize',FS); ylabel(ax2,'Value','FontSize',FS);
legend(ax2,'Q_{pos}','R','Location','best','FontSize',7,'Box','off');
title(ax2,'DDPG Agent  |  Live Q & R Adaptation','FontSize',10,'FontWeight','bold');

%% AX3 — ERROR
ax3=axes(f3,'Position',[0.09 0.13 0.88 0.82]); hold on;
h_er=animatedline(ax3,'Color',Craw,'LineWidth',LW,'DisplayName','Raw');
h_ef=animatedline(ax3,'Color',Cfix,'LineWidth',LW,'DisplayName','Fixed KF');
h_ek=animatedline(ax3,'Color',Crl, 'LineWidth',LW,'DisplayName','DDPG KF');
yline(ax3,0,'Color',[0.60 0.60 0.60],'LineWidth',0.5);
ylim(ax3,[-30 90]); xlim(ax3,[0 10]);
set(ax3,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',FS,...
    'YTick',-30:5:90,'XTick',0:0.5:10,'TickDir','in');
grid(ax3,'on');
xlabel(ax3,'Time (s)','FontSize',FS); ylabel(ax3,'Error (cm)','FontSize',FS);
legend(ax3,'Raw','Fixed KF','DDPG KF','Location','best','FontSize',7,'Box','off');
title(ax3,'Ranging Error','FontSize',10,'FontWeight','bold');

%% AX4 — RMSE
ax4=axes(f4,'Position',[0.09 0.13 0.88 0.82]); hold on;
h_rf=animatedline(ax4,'Color',Cfix,'LineWidth',LW,'DisplayName','Fixed KF');
h_rr=animatedline(ax4,'Color',Crl, 'LineWidth',LW,'DisplayName','DDPG KF');
ylim(ax4,[0 25]); xlim(ax4,[0 10]);
set(ax4,'Color',GS,'GridAlpha',0.10,'Box','on','FontSize',FS,...
    'YTick',0:1:25,'XTick',0:0.5:10,'TickDir','in');
grid(ax4,'on');
xlabel(ax4,'Time (s)','FontSize',FS); ylabel(ax4,'RMSE (cm)','FontSize',FS);
legend(ax4,'Fixed KF','DDPG KF','Location','best','FontSize',7,'Box','off');
title(ax4,'Running RMSE','FontSize',10,'FontWeight','bold');

%% AX5 — TRAJECTORY
ax5=axes(f5,'Position',[0.10 0.07 0.87 0.89]); hold on;
rectangle('Position',[0 0 W H],'EdgeColor',[0.75 0.75 0.75],'LineWidth',0.8,'LineStyle','--');
plot(ax5,true_x,true_y,'Color',[0.95 0.82 0.05],'LineWidth',0.7);
anc=[0 0;W 0;W H;0 H];
for a=1:4
    scatter(ax5,anc(a,1),anc(a,2),80,'r','s','filled');
    text(ax5,anc(a,1)+0.5,anc(a,2)+0.8,sprintf('A%d',a),...
        'FontSize',8,'FontWeight','bold','Color',[0.75 0.08 0.08]);
end
h_cov=plot(ax5,NaN,NaN,'Color',[0.85 0.10 0.08],'LineWidth',0.8);
h_dot=scatter(ax5,NaN,NaN,60,Crl,'o','filled');
h_pnk=gobjects(4,1);
for a=1:4
    h_pnk(a)=plot(ax5,[anc(a,1) NaN],[anc(a,2) NaN],...
        'Color',[1.0 0.42 0.70],'LineWidth',0.7);
end
xlim(ax5,[-2 W+3]); ylim(ax5,[-2 H+3]); axis(ax5,'equal');
set(ax5,'Color',GS,'GridAlpha',0.10,'FontSize',FS,'Box','on','TickDir','in');
grid(ax5,'on'); legend(ax5,'off');
xlabel(ax5,'X (m)','FontSize',FS); ylabel(ax5,'Y (m)','FontSize',FS);
title(ax5,'Boustrophedon Scan  |  4 Anchors','FontSize',10,'FontWeight','bold');

%% SERIAL MONITOR HEADER — Command Window
fprintf('%-6s %-8s %-11s %-11s %-11s %-11s %-10s %-10s %-10s\n',...
    '#','t(s)','True(m)','Raw(m)','FixKF(m)','DDPG(m)','Qpos','R','Err(cm)');
fprintf('%s\n',repmat('-',1,98));

%% ANIMATION
for k=1:N
    if ~all(ishandle([f1 f2 f3 f4 f5])); break; end

    addpoints(h_tr, t(k),true_range(k));
    addpoints(h_raw,t(k),raw(k));
    addpoints(h_fix,t(k),kf_fix(k));
    addpoints(h_rl, t(k),kf_rl(k));
    addpoints(h_qp, t(k),qp_hist(k));
    addpoints(h_rv, t(k),rv_hist(k));
    addpoints(h_er, t(k),err_raw(k));
    addpoints(h_ef, t(k),err_fix(k));
    addpoints(h_ek, t(k),err_rl(k));
    addpoints(h_rf, t(k),rmse_fix(k));
    addpoints(h_rr, t(k),rmse_rl(k));

    if t(k)>10
        xlim(ax1,[t(k)-10 t(k)]); xlim(ax2,[t(k)-10 t(k)]);
        xlim(ax3,[t(k)-10 t(k)]); xlim(ax4,[t(k)-10 t(k)]);
    end

    set(h_cov,'XData',true_x(1:k),'YData',true_y(1:k));
    set(h_dot,'XData',true_x(k),'YData',true_y(k));
    for a=1:4
        set(h_pnk(a),'XData',[anc(a,1) true_x(k)],'YData',[anc(a,2) true_y(k)]);
    end

    if mod(k,5)==0
        fprintf('%-6d %-8.2f %-11.5f %-11.5f %-11.5f %-11.5f %-10.6f %-10.6f %+.4f\n',...
            k,t(k),true_range(k),raw(k),kf_fix(k),kf_rl(k),...
            qp_hist(k),rv_hist(k),err_rl(k));
    end

    pause(0.08); drawnow limitrate;
end

fprintf('%s\n',repmat('-',1,98));
fprintf('Fixed KF RMSE : %.4f cm\n',rmse_fix(N));
fprintf('DDPG  KF RMSE : %.4f cm\n',rmse_rl(N));
fprintf('Improvement   : %.2f%%\n',(1-rmse_rl(N)/rmse_fix(N))*100);

%% EXPORT ALL RESULTS TO CSV
fprintf('\nExporting results to CSV...\n');

% Build table
csv_file = 'uwb_ddpg_results.csv';
fid = fopen(csv_file, 'w');

% Header
fprintf(fid, 'Sample,Time_s,TrueRange_m,RawUWB_m,FixedKF_m,DDPG_KF_m,');
fprintf(fid, 'Qpos,R,Error_Raw_cm,Error_FixKF_cm,Error_DDPG_cm,');
fprintf(fid, 'RMSE_FixKF_cm,RMSE_DDPG_cm,NLOS_Flag,TrueX_m,TrueY_m\n');

% Data rows
for k=1:N
    fprintf(fid, '%d,%.4f,%.6f,%.6f,%.6f,%.6f,', ...
        k, t(k), true_range(k), raw(k), kf_fix(k), kf_rl(k));
    fprintf(fid, '%.8f,%.8f,%.4f,%.4f,%.4f,', ...
        qp_hist(k), rv_hist(k), err_raw(k), err_fix(k), err_rl(k));
    fprintf(fid, '%.4f,%.4f,%d,%.4f,%.4f\n', ...
        rmse_fix(k), rmse_rl(k), double(nlos_mask(k)), true_x(k), true_y(k));
end
fclose(fid);

fprintf('Saved: %s  [%d rows x 16 columns]\n', csv_file, N);
fprintf('\nColumns: Sample | Time | TrueRange | Raw | FixKF | DDPG |\n');
fprintf('         Qpos | R | ErrRaw | ErrFix | ErrDDPG |\n');
fprintf('         RMSE_Fix | RMSE_DDPG | NLOS | TrueX | TrueY\n');