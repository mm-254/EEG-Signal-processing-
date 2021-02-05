%HS DATA
%Mandira Marambe
%Assessing the ability of EEG signals in discriminating between surface
%textures during grasp and lift tasks
%Using data produced by Luciw et.al.
%(https://www.nature.com/articles/sdata201447#Sec22)
%Surface series trials of subjects 7 and 11 are used for these projects

%______________________________DATA_________________________________________
%load data (non-windowed series)
% Must load and run two subjects separately

%Subject 7
Runs= WEEG_GetEventsInHS(7,2);
p_1 = load ('HS_P7_S1.mat');
p_2 = load ('HS_P7_S4.mat');
%Use all lifts structure for separating trials for training
all= load('P7_AllLifts.mat');
all = [all.P.AllLifts(1:34,:);all.P.AllLifts(97:130,:)];
% % 
% % % Subject 11
% Runs= WEEG_GetEventsInHS(11,2);
% p_1 = load ('HS_P11_S3.mat');
% p_2 = load ('HS_P11_S5.mat');
% all= load('P11_AllLifts.mat');
% all = [all.P.AllLifts(63:96,:);all.P.AllLifts(131:164,:)];
% % % % %________________Extract data______________________
series1 = p_1.hs.eeg.sig';
series2 = p_2.hs.eeg.sig';
% 
% 
fs = 500;
% % % % NumLifts = numel(all(:,1));
% % % 
all_trials = [];
silk_trials=[];
suede_trials=[];
sandpaper_trials=[];
% % % 
% % % % %__________________________PREPROCESSING_____________________________________
% % % % 
% % % % for trials = 1: NumLifts
% % %     
% %     all_trials = [];
% %     
%     %________________Preprocessing_______________________
    %Mastoid channels are channels 17 and 22
    %Re-reference EEG data from mastoid channels as signal of interest is
    %in motor cortex
    
    % find mean between channels
        avg = mean([series1(17,:);series1(22, :)]);
        avg2 = mean([series2(17,:);series2(22, :)]);
        
    %Select six channels of interest and discard the rest
        series1([1:12, 16:23, 27:32], :) = [];
        series2([1:12, 16:23, 27:32], :) = [];
         

    % subtract mean 
        mean_matrix = repmat(avg, size(series1,1),1);
        reref_series1 = series1 - mean_matrix;
        mean_matrix2 = repmat(avg2, size(series2,1),1);
        reref_series2 = series2 - mean_matrix2;
     
    %Artifact rejection and bandpass filter
    h = fir1(1000, [1/(fs/2) 12/(fs/2)]);
       
    for i = 1: size(reref_series1,1)   
    %Filtering between 7-30Hz to remove most of the noise
        reref_series1(i,:) = filtfilt(h, 1, double(reref_series1(i,:)));
    
    %DC baseline correction
    %First 1s (500 samples) used because LED turns on after 2s
        chan_signal = reref_series1(i,:);
        baseline_avg = mean(chan_signal);
        reref_series1(i,:) = (reref_series1(i,:)-baseline_avg)/std(chan_signal);
    end
    
        for i = 1: size(reref_series2,1)   
    %Filtering between 7-30Hz to remove most of the noise
        reref_series2(i,:) = filtfilt(h, 1, double(reref_series2(i,:)));
    
    %DC baseline correction
    %First 1s (500 samples) used because LED turns on after 2s
        chan_signal = reref_series2(i,:);
        baseline_avg = mean(chan_signal);
        reref_series2(i,:) = (reref_series2(i,:)-baseline_avg)/std(chan_signal);
    end
    
    %epoching and event extraction
    contact_time1 = Runs.Events(1).tTouch;
    contact_samps1 = round(contact_time1.*fs);
    contact_time2 = Runs.Events(2).tTouch;
    contact_samps2 = round(contact_time2.*fs);
     
for t = 1:size(contact_samps1, 1)         
           trial1 = reref_series1(:,contact_samps1(t):contact_samps1(t) + 1500);
           all_trials = [all_trials; mean(trial1)];
end

for t = 1:size(contact_samps2, 1)         
           trial2 = reref_series2(:,contact_samps2(t):contact_samps2(t) + 1500);
           all_trials = [all_trials;mean(trial2)];
end

%Collect all trial epochs; separate surfaces for training
for trial = 1:size(all_trials,1)
        if all(trial,5) ==1
            silk_trials = [silk_trials;all_trials(trial,:)];
        elseif all(trial,5) ==2
             suede_trials = [suede_trials;all_trials(trial,:)];
        elseif all(trial,5) ==3
              sandpaper_trials = [sandpaper_trials;all_trials(trial,:)];
        end
end
% 
% %___________________________DIMENSIONALITY REDUCTION____________________________________
% 
% % %Downsampling
% % %Downsampling to 50Hz as frequencies of interest are below 25Hz
% %  reduced_trials_silk = downsample(silk_trials',10);
% %  reduced_trials_silk= reduced_trials_silk';
% %  reduced_trials_suede = downsample(suede_trials',10);
% %  reduced_trials_suede= reduced_trials_suede';
% %  reduced_trials_sandpaper = downsample(sandpaper_trials',10);
% %  reduced_trials_sandpaper= reduced_trials_sandpaper';
% %  
% %  %Time window of 
% % % reduced_trials_silk=  reduced_trials_silk(:,75:115);
% % % reduced_trials_suede=  reduced_trials_suede(:,75:115);
% % % reduced_trials_sandpaper=  reduced_trials_sandpaper(:,75:115);
% % % reduced_trials =[reduced_trials;reduced_trial];
% % 
% % %PCA
% % 
% % % silk
% % C_silk = cov(reduced_trials_silk); 
% % 
% % [V_silk, D_silk] = eigs(double(C_silk));
% % q1_silk = V_silk(:, 1); 
% % q2_silk = V_silk(:, 2); 
% % 
% % PCA_projection_silk = [q1_silk, q2_silk]; 
% % PCA_rep_silk = reduced_trials_silk * PCA_projection_silk; 
% % 
% % 
% % % suede
% % C_suede = cov(reduced_trials_suede); 
% % 
% % [V_suede, D_suede] = eigs(double(C_suede));
% % q1_suede = V_suede(:, 1); 
% % q2_suede = V_suede(:, 2); 
% % 
% % PCA_projection_suede = [q1_suede, q2_suede]; 
% % PCA_rep_suede = reduced_trials_suede * PCA_projection_suede; 
% % 
% % % sandpaper
% % C_sp = cov(reduced_trials_sandpaper); 
% % 
% % [V_sp, D_sp] = eigs(double(C_sp));
% % q1_sp = V_sp(:, 1); 
% % q2_sp = V_sp(:, 2); 
% % 
% % PCA_projection_sp = [q1_sp, q2_sp]; 
% % PCA_rep_sp = reduced_trials_sandpaper * PCA_projection_sp; 
% % 
% % % plot
% % scatter(PCA_rep_silk(:, 1), PCA_rep_silk(:, 2))
% % hold on
% % scatter(PCA_rep_suede(:, 1), PCA_rep_suede(:, 2))
% % hold on
% % scatter(PCA_rep_sp(:, 1), PCA_rep_sp(:, 2))
% % legend('Silk', 'Suede', 'Sandpaper')
% % title('Principal Components of Surface data')
% % xlabel('PC 1')
% % ylabel('PC 2')
% 
% 
% % %________________________________SPECTROGRAMS__________________________________
% % 
% frequency bins 
f = 8:0.2:13;
timevector = 0:1/fs:3;
T2W =   2; % Time-bandwidth product
% The code automatically uses 2TW - 1 tapers
n_cycles = 5; % Wavelet

spec_silk=zeros(size(f,2),size(all_trials,2));
ersp_t1 = zeros(size(silk_trials,1), size(all_trials,2));
ersp_f1 =zeros(size(silk_trials,1),size(f,2));
for trial = 1:size(silk_trials,1)
[S, ~, ~, t_silk] = mttfr(silk_trials(trial,:)', fs, f, n_cycles, T2W, 0);
spec_silk = spec_silk +pow2db(S);
ersp_t1(trial,:) = mean(pow2db(S(1:12,:)));
ersp_f1(trial,:) = mean(pow2db(S(:,[1*fs:2*fs])),2)';
end
spec_silk = spec_silk./size(silk_trials,1);
meanersp_t1 = mean(spec_silk(1:12,:)); %average across frequencies 8-11Hz
meanersp_f1 = mean(spec_silk(:,[1*fs:2*fs]),2)'; %average across time

spec_suede=zeros(size(f,2),size(all_trials,2));
ersp_t2 = zeros(size(suede_trials,1), size(all_trials,2));
ersp_f2 =zeros(size(suede_trials,1),size(f,2));
for trial = 1:size(suede_trials,1)
[S, ~, ~, t_suede] = mttfr(suede_trials(trial,:)', fs, f, n_cycles, T2W, 0);
spec_suede = spec_suede +pow2db(S);
ersp_t2(trial,:) = mean(pow2db(S(1:12,:)));
ersp_f2(trial,:) = mean(pow2db(S(:,[1*fs:2*fs])),2)';
end
spec_suede = spec_suede./size(suede_trials,1);
meanersp_t2 = mean(spec_suede(1:12,:));
meanersp_f2 = mean(spec_suede(:,[1*fs:2*fs]),2);

spec_sandpaper=zeros(size(f,2),size(all_trials,2));
ersp_t3 = zeros(size(sandpaper_trials,1), size(all_trials,2));
ersp_f3 =zeros(size(sandpaper_trials,1),size(f,2));
for trial = 1:size(sandpaper_trials,1)
[S, ~, ~, t_sandpaper] = mttfr(sandpaper_trials(trial,:)', fs, f, n_cycles, T2W, 0);
spec_sandpaper = spec_sandpaper +pow2db(S);
ersp_t3(trial,:) = mean(pow2db(S(1:12,:)));
ersp_f3(trial,:) = mean(pow2db(S(:,[1*fs:2*fs])),2)';
end
spec_sandpaper = spec_sandpaper./size(sandpaper_trials,1);
meanersp_t3 = mean(spec_sandpaper(1:12,:));
meanersp_f3 = mean(spec_sandpaper(:,[1*fs:2*fs]),2);
% 
% Plot spectrograms
plottfr(t_silk, f, spec_silk); 
% plot(f_silk, spec_silk); 
% hold on;
title("Mean Spectrogram, Silk");
plottfr(t_suede, f, spec_suede);
% plot(f_suede, spec_suede); 
% hold on;
title("Mean Spectrogram, Suede");
plottfr(t_sandpaper, f, spec_sandpaper); 
% plot(f_sandpaper, spec_sandpaper);
title("Mean Spectrogram, Sandpaper");
% legend("Silk","Suede","Sandpaper")
% xlabel("Frequency (Hz)")
% ylabel ("Power(dB)")
% % 
%Plot ERSP_T
figure;
plot(timevector, ersp_t1,'r');
hold on;
plot(timevector, ersp_t2,'k');
hold on;
plot(timevector, ersp_t3,'k');
title("ERSP, time axis: Silk vs Not Silk")
% title("Mean ERSP, time axis")
% legend("Silk","Suede","Sandpaper")
xlabel("Time (s)")
ylabel("Power (dB)")

%Plot ERSP_F
figure;
plot(f, ersp_f1,'r');
hold on;
plot(f, ersp_f2,'k');
hold on;
plot(f, ersp_f3,'k');
title("ERSP, frequency axis: Silk vs Not Silk")
% title("Mean ERSP, frequency axis")
% legend("Silk","Suede","Sandpaper")
xlabel("Frequency (Hz)")
ylabel("Power (dB)")


% %______________________NONPARAMETRIC PERMUTATION TEST______________________
% 
% % %True p-value = 0.33
silkf1 = ersp_f1';
not_silkf1 = [ersp_f2', ersp_f3'];
silk = ersp_t1';
not_silk = [ersp_t2', ersp_t3'];
% 
% %Using a function permutest found on Matlab website
[c_f,p_f] = permutest(silk, not_silk, false,0.05,1000); %1000 permutations
[c_t,p_t] = permutest(silk, not_silk, false,0.05,1000); %1000 permutations

%_____________________________SVM CLASSIFIER_____________________________________

% % %Classify 'silk' and 'not silk'

% Build and train SVM
X_train = [silk(1297:end,1:15)';silk2(1297:end,1:15)';not_silk(1297:end,1:30)';not_silk2(1297:end,1:30)'];
%  X_train = [silk2(1297:end,1:15)';not_silk2(1297:end,1:30)'];
%  X_train = [silkf1(:,1:15)';silkf(:,1:15)';not_silkf1(:,1:30)';not_silkf(:,1:30)'];
% % % 
% % % %Train the SVM Classifier
y_train = ones(size(X_train,1),1);
y_train(31:end) = -1;
model = fitcsvm(X_train, y_train);
% % 
% Plot training data
% plot(X_train(1:30, 1), X_train(1:30, 2), '*k',...
%     'linew', 2, 'markersize', 10);
% hold on;
% plot(X_train(31:end, 1), X_train(31:end, 2), 'or',...
%     'linew', 2, 'markersize', 10);
% xlabel('F1 (Hz)', 'fontsize', 14);
% ylabel('F2 (Hz)', 'fontsize', 14);
% set(gca, 'fontsize', 14);
% 
% % Use remaining data to test
X_test = [silk(1297:end,16:end)';silk2(1297:end,16:end)';not_silk(1297:end,31:end)';not_silk2(1297:end,31:end)'];
% X_test = [silk2(1297:end,16:end)';not_silk2(1297:end,31:end)'];
% X_test = [silkf1(:,16:end)';silkf(:,16:end)';not_silkf1(:,31:end)';not_silkf(:,31:end)'];
% % % 
% % % % Test the classifier
result = model.predict(X_test);
% 
% % from true labels
y_test = ones(size(X_test,1), 1);
y_test(16:end) = -1;
% 
accuracy = sum(result == y_test) * 100/ numel(y_test);
fprintf(1, 'Test accuracy is %0.2f %% ...\n', accuracy);

result_train = model.predict(X_train);
accuracy_train = sum(result_train == y_train) * 100/ numel(y_train);
fprintf(1, 'Training accuracy is %0.2f %% ...\n', accuracy_train);

FalseAlarmRate = sum(result(16:end) == 1) * 100/ numel(result(16:end));
fprintf(1, 'FA Rate is %0.2f %% ...\n', FalseAlarmRate);

MissRate = sum(result(1:15) == -1) * 100/ numel(result(1:15));
fprintf(1, 'Miss Rate is %0.2f %% ...\n', MissRate);
% 
HitRate = 100-MissRate;
fprintf(1, 'Hit Rate is %0.2f %% ...\n', HitRate);
