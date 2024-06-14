clear all

addpath(genpath('C:\Users\tanabes\Documents\npy-matlab-master\npy-matlab-master'))

N = 3000;
h = 1; %ms 
n_trials_load = 50 ;

cd E:\20191019_Michigan_Comp_NSC\DePasquale_Abbott_2016_arXiv
J = readNPY(['fig3_h' num2str(h*1000) 'us_' num2str(n_trials_load) '_J.npy']);
J_f = readNPY(['fig3_h' num2str(h*1000) 'us_' num2str(n_trials_load) '_J_f.npy']);
U = readNPY(['fig3_h' num2str(h*1000) 'us_' num2str(n_trials_load) '_U.npy']);
W = readNPY(['fig3_h' num2str(h*1000) 'us_' num2str(n_trials_load) '_W.npy']);

n_trials = 100;

% seq type, in, out
% 1, F F, F
% 2, F T, T
% 3, T F, T
% 4, T T, F

F_in = []; F_out = [];
F_seq_onset = [];
ini_seg = zeros(1,floor((300/h)*rand));
F_in = ini_seg; F_out = ini_seg; 
F_seq_onset = [ini_seg; ini_seg; ini_seg; ini_seg];
for tr = 1:n_trials
    seq_type = randi(4);
    inter_trial = floor((100+300*rand)/h);
    if seq_type == 1
       F_in = [F_in [ones(1,floor(100/h))*0.3 zeros(1,floor(300/h)) ones(1,floor(100/h))*0.3 zeros(1,floor((300+500)/h)) zeros(1,inter_trial)]];
       F_out = [F_out [zeros(1,floor((100+300+100+300)/h)) -sin((2*pi)*(0:(h/1000):(0.5-(h/1000)))) zeros(1,inter_trial)]];
       
       tmp_ind = zeros(4,floor((500)/h)); tmp_ind(1,1) = 1;
       F_seq_onset = [F_seq_onset [zeros(4,floor((100+300+100+300)/h)) tmp_ind zeros(4,inter_trial)]];
    elseif seq_type == 2
       F_in = [F_in [ones(1,floor(100/h))*0.3 zeros(1,floor(300/h)) ones(1,floor(300/h))*0.3 zeros(1,floor((300+500)/h)) zeros(1,inter_trial)]];
       F_out = [F_out [zeros(1,floor((100+300+300+300)/h)) sin((2*pi)*(0:(h/1000):(0.5-(h/1000)))) zeros(1,inter_trial)]];
       
       tmp_ind = zeros(4,floor((500)/h)); tmp_ind(2,1) = 1;
       F_seq_onset = [F_seq_onset [zeros(4,floor((100+300+100+300)/h)) tmp_ind zeros(4,inter_trial)]];
    elseif seq_type == 3
       F_in = [F_in [ones(1,floor(300/h))*0.3 zeros(1,floor(300/h)) ones(1,floor(100/h))*0.3 zeros(1,floor((300+500)/h)) zeros(1,inter_trial)]];
       F_out = [F_out [zeros(1,floor((300+300+100+300)/h)) sin((2*pi)*(0:(h/1000):(0.5-(h/1000)))) zeros(1,inter_trial)]];
       
       tmp_ind = zeros(4,floor((500)/h)); tmp_ind(3,1) = 1;
       F_seq_onset = [F_seq_onset [zeros(4,floor((100+300+100+300)/h)) tmp_ind zeros(4,inter_trial)]];
    elseif seq_type == 4
        F_in = [F_in [ones(1,floor(300/h))*0.3 zeros(1,floor(300/h)) ones(1,floor(300/h))*0.3 zeros(1,floor((300+500)/h)) zeros(1,inter_trial)]];
       F_out = [F_out [zeros(1,floor((300+300+300+300)/h)) -sin((2*pi)*(0:(h/1000):(0.5-(h/1000)))) zeros(1,inter_trial)]];
       
       tmp_ind = zeros(4,floor((500)/h)); tmp_ind(4,1) = 1;
       F_seq_onset = [F_seq_onset [zeros(4,floor((100+300+100+300)/h)) tmp_ind zeros(4,inter_trial)]];
    end
end

% figure
% subplot(2,1,1)
% plot(F_in); ylim([0 0.4]);ylabel('F_{in}')
% subplot(2,1,2)
% plot(F_out)
% xlabel('time(ms)')
% ylabel('F_{out}')
% xlim([0 9000])

% figure
% imagesc(F_seq_onset)

T = length(F_in)*h; %ms
t = 0:h:T; firings = [];


%% spiking network, validate

N_out = 1;
N_in  = 1;
mu = -40;
g_f = 12;
g = 10 ;
I = 10;
tau_s = 100;
tau_f = 2;
tau_m = 20;
V_rest = -65;
V_th = -55;
V = ones(N,1)*V_rest + 10*rand(N,1);
s = rand(N,1);
f = rand(N,1);

s_time = NaN(N,length(t)-1);
V_dot =@(V,s,f,F_in_i)((V_rest-V+g*(J*s+J_f*f+U*F_in_i)+I)/tau_m);
s_dot =@(s)(-s/tau_s);
f_dot =@(f)(-f/tau_f);
for i = 1:(length(t)-1)
    fired = find(V >= V_th); %%%
    firings = [firings; t(i)+0*fired,fired];
    V(fired)=V_rest;
    s(fired)=s(fired)+1;
    f(fired)=f(fired)+1;
    V = V + h*V_dot(V,s,f,F_in(i));
    s = s + h*s_dot(s);
    f = f + h*f_dot(f);
    
    s_time(:,i) = s;
    
    if mod(i, 100) == 0
        disp(['validate ' num2str(t(i)) 'ms ' num2str(length(firings)) ' fired'])
    end
end
Ws = s_time'*W;

%% check
% figure
% plot(F_out)
% hold on
% plot(Ws)

% figure
% plot(firings(:,1),firings(:,2),'.')
% xlabel('time(ms)');ylabel('neuron #')
% xlim([0 9000])

%% save

save(['fig3_h' num2str(h*1000) 'us_' num2str(n_trials_load) '_Ws_validation_' num2str(n_trials) 'trials'],'Ws') 
save(['fig3_h' num2str(h*1000) 'us_' num2str(n_trials_load) '_F_seq_onset_validation_' num2str(n_trials) 'trials'],'F_seq_onset') 
%% statistics

% seq type, in, out
% 1, F F, F
% 2, F T, T
% 3, T F, T
% 4, T T, F
delay = 0

Seq1_ind = find(F_seq_onset(1,:));
Seq1_mn = NaN(1,length(Seq1_ind));
Seq1_cat = NaN((500+delay),length(Seq1_ind));
for i = 1:length(Seq1_ind)
 ind = Seq1_ind(i);
 Seq1_mn(i) = median(Ws((ind+floor((400)/h)-1)));
 Seq1_cat(:,i) = Ws(ind-delay:(ind+floor((500)/h)-1)+delay);
end
Seq2_ind = find(F_seq_onset(2,:));
Seq2_mn = NaN(1,length(Seq2_ind));
Seq2_cat = NaN((500+delay),length(Seq2_ind));
for i = 1:length(Seq2_ind)
 ind = Seq2_ind(i);
 Seq2_mn(i) = median(Ws((ind+floor((400)/h)-1)));
 Seq2_cat(:,i) = Ws(ind-delay:(ind+floor((500)/h)-1+delay));
end
Seq3_ind = find(F_seq_onset(3,:));
Seq3_mn = NaN(1,length(Seq3_ind));
Seq3_cat = NaN((500+delay),length(Seq3_ind));
for i = 1:length(Seq3_ind)
 ind = Seq3_ind(i);
 Seq3_mn(i) = median(Ws((ind+floor((400)/h)-1)));
 Seq3_cat(:,i) = Ws(ind-delay:(ind+floor((500)/h)-1+delay));
end
Seq4_ind = find(F_seq_onset(4,:));
Seq4_mn = NaN(1,length(Seq4_ind));
Seq4_cat = NaN((500+delay),length(Seq4_ind));
for i = 1:length(Seq4_ind)
 ind = Seq4_ind(i);
 Seq4_mn(i) = median(Ws((ind+floor((400)/h)-1)));
 Seq4_cat(:,i) = Ws(ind-delay:(ind+floor((500)/h)-1+delay));
end

response_T = [Seq2_mn Seq3_mn];
response_F = [Seq1_mn Seq4_mn];

response_T_cat = [Seq2_cat Seq3_cat];
response_F_cat = [Seq1_cat Seq4_cat];


p3 = ranksum(response_T, response_F)

figure
Seq_x = [Seq1_mn'; Seq2_mn'; Seq3_mn'; Seq4_mn'];
Seq_g = [zeros(length(Seq1_mn), 1); ones(length(Seq2_mn), 1); 2*ones(length(Seq3_mn), 1); 3*ones(length(Seq4_mn), 1)];
boxplot(Seq_x, Seq_g)

figure('Renderer', 'painters', 'Position', [10 10 160 200])
h1 = scatter(ones(length(response_T),1),response_T, 10,'k', 'filled'); 
hold on
m1 = plot(1,mean(response_T),'+r')
hold on
h2 = scatter(ones(length(response_F),1)*2,response_F, 10,'k', 'filled'); 
hold on
m2 = plot(2,mean(response_F),'+r')
set(gca,'TickDir','out');
xlim([0 3]);
ylabel('mean response')
set(gca,'XTick',[1 2],'XTickLabel',{'T';'F'})
title(['p=' num2str(p3)])

figure
plot(median(response_T_cat,2))
hold on
plot(median(response_F_cat,2))


figure
plot(median(Seq1_cat,2),'Color',[0.6350 0.0780 0.1840])
hold on
plot(median(Seq2_cat,2),'Color',[0 0.4470 0.7410])
hold on
plot(median(Seq3_cat,2),'Color',[0.3010 0.7450 0.9330])
hold on
plot(median(Seq4_cat,2),'Color',[0.8500 0.3250 0.0980])
ylabel('Ws') %xlabel('time(ms)');
legend('T,T then F','F,T then F','T,F then T','F,F then F', 'location', 'best')

figure
plot(sin((2*pi)*(0:(h/1000):(0.5-(h/1000)))),'Color',[0.6350 0.0780 0.1840])
hold on
plot(-sin((2*pi)*(0:(h/1000):(0.5-(h/1000)))),'Color',[0 0.4470 0.7410])
ylim([-1.5 1.5])
ylabel('ideal reponse')
xlabel('time(ms)')






