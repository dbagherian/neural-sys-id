
binthresh = 0.1;

load('pushpull_trainingdata.mat')
b1g_true = 0.1*bip1_gc_syn(:, 8);
b2g_true = 0.1*bip2_gc_syn(:, 8);
b11am3_true = bip11_am3_syn;
am3g_true = squeeze(am3_gc_syn(:, 8));

b1g_true_bin = single(b1g_true>0);
b2g_true_bin = single(b2g_true>0);
b11am3_true_bin = single(b11am3_true>0);
am3g_true_bin = single(am3g_true>0);

%%
sds = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 95000];

sys_scores = zeros(1, length(sds));
supp_scores = zeros(1, length(sds));


for sd_i = 1:length(sds)
    sd = sds(sd_i);
    myfile=['C:\Users\Dawna\Desktop\mousesimulations\pushpull\KDD_PushPull_simtraining_sd', num2str(sd), '.mat'];
    load(myfile);

    b1g_lrnd = abs(bip1_gc_syn_hist(end, :)); 
    b2g_lrnd = abs(bip2_gc_syn_hist(end, :));
    b11am3_lrnd = abs(squeeze(bip11_am3_syn_hist(end, :, :)));
    am3g_lrnd = abs(am3_gc_syn_hist(end, :));

    for i =1:21
        if am3g_lrnd(i)<binthresh
            b11am3_lrnd(:, i)=0; %if amacrine cell is not connected to gc, its input is irrelevant.
        end
    end



    b1g_lrnd_bin = single(b1g_lrnd>binthresh);
    b2g_lrnd_bin = single(b2g_lrnd>binthresh);
    b11am3_lrnd_bin = single(b11am3_lrnd>binthresh);
    am3g_lrnd_bin = single(am3g_lrnd>binthresh);

    sysnum = sum(b1g_lrnd'.*b1g_true) + sum(b2g_lrnd' .* b2g_true) + sum(am3g_lrnd .* am3g_true') + sum(sum(b11am3_lrnd .* b11am3_true));
    sysdenom1 = (b1g_lrnd* b1g_lrnd') + (b2g_lrnd * b2g_lrnd') + (am3g_lrnd * am3g_lrnd') + sum(sum(b11am3_lrnd .* b11am3_lrnd));
    sysdenom2 = (b1g_true'* b1g_true) + (b2g_true' * b2g_true) + (am3g_true' * am3g_true) + sum(sum(b11am3_true .* b11am3_true));

    sys = 2*sysnum/(sysdenom1 + sysdenom2);
    sys_scores(sd_i) = sys;

    suppnum = sum(b1g_lrnd_bin'.*b1g_true_bin) + sum(b2g_lrnd_bin' .* b2g_true_bin) + sum(am3g_lrnd_bin .* am3g_true_bin') + sum(sum(b11am3_lrnd_bin .* b11am3_true_bin));
    suppdenom1 = (b1g_lrnd_bin* b1g_lrnd_bin') + (b2g_lrnd_bin * b2g_lrnd_bin') + (am3g_lrnd_bin * am3g_lrnd_bin') + sum(sum(b11am3_lrnd_bin .* b11am3_lrnd_bin));
    suppdenom2 = (b1g_true_bin'* b1g_true_bin) + (b2g_true_bin' * b2g_true_bin) + (am3g_true_bin' * am3g_true_bin) + sum(sum(b11am3_true_bin .* b11am3_true_bin));

    supp = 2*suppnum/(suppdenom1 + suppdenom2);
    supp_scores(sd_i) = supp;
    
end

save('scores.mat', 'supp_scores', 'sys_scores');