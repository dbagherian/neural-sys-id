load('F:\Folder\Circuit2_Training_Data.mat')

thefolder ='C:\Path\To\Your\Folder\'; %Folder where trained models are stored



datass = {'60', '340', '700', '1400', '2800', '5500', '11000'};


seeds = {'10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000', '95000' };

kernelss = { '3', '8', '16', '28' };
kers = [ 3,8,16, 28];
knwls = {'0', '0', '10', '100', '200'};

system_all = {};
support_all = {};

goodams = zeros(5, 5);
for i =1:5
    for j=1:5
        a=squeeze(max(max(max(bip_am_syn(1, :, :, :, i, j)))));
        if a>0
            goodams(i, j)=1;
        end
    end
end
            

train_loss_all = zeros(length(kers), length(datass));
test_loss_all = zeros(length(kers), length(datass));
freenum=zeros(length(datass), length(kers), length(seeds));
lambdas = {};
learn_rates = {};
for data_i = 1:length(datass)
    for kernel_i = 1:length(kers)
        learn_rates{kernel_i, data_i}=[];
        lambdas{kernel_i, data_i}=[];
        support_all{kernel_i, data_i}=[];
        system_all{kernel_i, data_i}=[];
        for sd_i = 1:length(seeds)
        
        datas=datass{data_i};
        kernels = kernelss{kernel_i};
        knwl = knwls{kernel_i};
        kk = kers(kernel_i);

            exper = ['3lyrskp_randstim_data', datas, '_kernel', kernels, 'sd', seeds{sd_i}, '_intracellular_20210119_nosignconstraint'];
           thefile = [thefolder, exper, '.mat'];
           if isfile(thefile)
                load(thefile)
           else
                continue
           end

       
       train_loss_all(kernel_i, data_i)=train_loss_hist(end);
       test_loss_all(kernel_i, data_i)=test_loss_hist(end);
       
        true_ba = zeros(10, 10, kers(kernel_i), 5, 5);
        true_ba(:, :, 1:3, :, :) = squeeze(bip_am_syn);
        
        true_bg = zeros(10, 10, kers(kernel_i));
        true_bg(:, :, 1:3, :, :) = squeeze(bip_gc_syn);
        
        true_ba1= reshape(true_ba, [kk*100, 25]);
        [bipinds, aminds] = find(true_ba1 > 0); 
        
        true_ag = squeeze(am_gc_syn).*goodams;
        true_ag1 = reshape(true_ag, [25, 1]);
        
        lrnd_ba = squeeze((bip_am_syn_hist(end, :, :, :, :, :)));
        lrnd_ba1= reshape(lrnd_ba, [kk*100, 25]);
        [maxx, maxinds]=max(lrnd_ba1,[], 2);
        newinds = maxinds(bipinds);

        
        
        lrnd_bg = squeeze((bip_gc_syn_hist(end, :, :, :)));
        
        lrnd_bg1=reshape(lrnd_bg, [10, kk*10]);
        
        lrnd_ag = squeeze((am_gc_syn_hist(end, :, :)));
        lrnd_ag1 = reshape(lrnd_ag, [25, 1]);
        
        binthresh =0.07;
        lrnd_ag_bin = lrnd_ag>binthresh;
        lrnd_ba_bin = lrnd_ba>binthresh;
        lrnd_bg_bin = lrnd_bg>binthresh;
        lrnd_ag1_bin = lrnd_ag1>binthresh;
        lrnd_ba1_bin = lrnd_ba1>binthresh;
        lrnd_bg1_bin = lrnd_bg1>binthresh;
        
       true_ag_bin = true_ag>0;
        true_ba_bin = true_ba>0;
        true_bg_bin = true_bg>0;
        true_ag1_bin = true_ag1>0;
        true_ba1_bin = true_ba1>0;

        
        norm_factor = sum(sum(sum(sum(sum(true_ba.^2)))))+ sum(sum(sum(true_ag.^2))) + sum(sum(sum(true_bg.^2)));
        norm_factor = norm_factor+ sum(sum(sum(sum(sum(lrnd_ba.^2)))))+ sum(sum(sum(lrnd_ag.^2))) + sum(sum(sum(lrnd_bg.^2)));
        norm_factor = 0.5*norm_factor;
        
        
        norm_factor_bin = sum(sum(sum(sum(sum(true_ba_bin.^2)))))+ sum(sum(sum(true_ag_bin.^2))) + sum(sum(sum(true_bg_bin.^2)));
        norm_factor_bin = norm_factor_bin+ sum(sum(sum(sum(sum(lrnd_ba_bin.^2)))))+ sum(sum(sum(lrnd_ag_bin.^2))) + sum(sum(sum(lrnd_bg_bin.^2)));
        norm_factor_bin = 0.5*norm_factor_bin;
        
        struct_proj1 = sum(sum(sum(sum(sum(true_ba1.*lrnd_ba1)))))+ sum(sum(sum(true_ag1.*lrnd_ag1))) + sum(sum(sum(true_bg.*lrnd_bg)));
        
        struct_proj1 = struct_proj1/norm_factor;
        
        struct_proj1_bin = sum(sum(sum(sum(sum(true_ba1_bin.*lrnd_ba1_bin)))))+ sum(sum(sum(true_ag1_bin.*lrnd_ag1_bin))) + sum(sum(sum(true_bg_bin.*lrnd_bg_bin)));
        
        struct_proj1_bin = struct_proj1_bin/norm_factor_bin;
        
        support_all{kernel_i, data_i}=cat(1, support_all{kernel_i, data_i}, struct_proj1_bin);
        system_all{kernel_i, data_i}=cat(1, system_all{kernel_i, data_i}, struct_proj1);
        lambdas{kernel_i, data_i}=cat(1, lambdas{kernel_i, data_i}, lambda);
        learn_rates{kernel_i, data_i}=cat(1, learn_rates{kernel_i, data_i}, learn_rate);
        
        
        bafree = sum(sum(sum(sum(sum(squeeze(abs(bip_am_syn_hist(1, :, :, :, :, :)))>0)))));
        agfree = sum(sum(sum(abs(am_gc_syn_hist(1, :, :))>0)));
        bgfree = sum(sum(sum(sum(abs(bip_gc_syn_hist(1, :, :, :))>0))));
        freenum(data_i, kernel_i, sd_i)=agfree+bafree+bgfree;
        end
    end
    

    
end
freenumavg=mean(mean(freenum, 3), 1);


stds = zeros(length(kers), length(datass));
mns = zeros(length(kers), length(datass));
for data_i = 1:length(datass)
    for kernel_i = 1:length(kers)
       mns(kernel_i, data_i) = mean(system_all{kernel_i, data_i});
       stds(kernel_i, data_i) = std(system_all{kernel_i, data_i});
    end
end

stds_struct = zeros(length(kers), length(datass));
mns_struct = zeros(length(kers), length(datass));
for data_i = 1:length(datass)
    for kernel_i = 1:length(kers)
       mns_struct(kernel_i, data_i) = mean(support_all{kernel_i, data_i});
       stds_struct(kernel_i, data_i) = std(support_all{kernel_i, data_i});
    end
end


colorbase = [1, 0.2, 0.2];
colors = zeros(6, 3);
for i =1:5
    colors(i, :) = colorbase*(i/5);
end



fig=figure; 
subplot(1, 2, 1);hold on;
for i =1:length(kers) 
    errorbar(log10([60, 340, 700, 1400, 2800, 5500, 11000]), mns_struct(i, :), stds_struct(i, :), '-o', 'LineWidth', 2.0, 'Color', squeeze(colors(i, :)))

end
legend(num2str(round(freenumavg')), [70, 250, 0, 0])
xlabel('Training data (log(# images))')
ylabel('Support recovery score')
ylim([0, 1]);
set(gca, 'FontSize', 16); hold off;

subplot(1, 2, 2)
hold on;
for i =1:length(kers)
    errorbar(log10([60, 340, 700, 1400, 2800, 5500, 11000]), mns(i, :), stds(i, :), '-o', 'LineWidth', 2.0, 'Color', squeeze(colors(i, :)))

end
legend(num2str(round(freenumavg')), [781, 286, 10, 0])
xlabel('Training data (log(# images))')
ylabel('System recovery score')
ylim([0, 1]);
set(gca, 'FontSize', 16);
hold off;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 18, 5], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
saveas(fig, fullfile(thefolder, sprintf('%s.png','struct_sys')))

fig=figure; hold on;
for i=1:length(kers)
plot(log10([60, 340, 700, 1400,2800, 5500, 11000]), train_loss_all(i, :), '-o', 'LineWidth', 2.0, 'Color', squeeze(colors(i, :)));
end
legend(num2str(round(freenumavg')))
xlabel('Training data (log(# images))')
ylabel('training loss')
set(gca, 'FontSize', 16);
set(gca, 'YScale', 'log');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 9], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
saveas(fig, fullfile(thefolder, sprintf('%s.png','final_train_loss')))

fig=figure;hold on;
for i=1:length(kers)
plot(log10([60, 340, 700, 1400, 2800, 5500, 11000]), test_loss_all(i, :), '-o', 'LineWidth', 2.0, 'Color', squeeze(colors(i, :)));
end
legend(num2str(round(freenumavg')))
xlabel('Training data (log(# images))')
ylabel('test loss')
set(gca, 'FontSize', 14);
set(gca, 'YScale', 'log');
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 12, 9], 'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125])
saveas(fig, fullfile(thefolder, sprintf('%s.png','final_test_loss')))


