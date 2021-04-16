# # generating annotation and ldsc scores for negative files
# 
# #for j in {'progen','basophil','cd4','cd8','dendritic','mono','nkCells','B_cells'}
# # do
# #for i in {1..22}
# do
# 
# 
#     python ldsc/make_annot.py --bed-file ../scate_files/${j}/big_peaks_centered-GCneg.bed --bimfile /home/eramamur/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${i}.bim --annot-file ${j}/GCneg.${i}.annot
# 
#     python ldsc/ldsc.py --l2 --bfile /home/eramamur/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${i} --ld-wind-cm 1 --annot ${j}/GCneg.${i}.annot --thin-annot --out ${j}/GCneg.${i} --print-snps /home/eramamur/resources/hapmap3_snps/hm.${i}.snp
# 
# done
# done
# 
# # Jansen and kunkle GWAS ldsc
python ldsc/ldsc.py --h2-cts /home/eramamur/resources/gwas/Jansen_GWAS.sumstats.gz --ref-ld-chr /home/eramamur/resources/1000G_EUR_Phase3_baseline/baseline. --out jansen_enrichments --ref-ld-chr-cts enrichment.ldcts --w-ld-chr /home/eramamur/resources/weights_hm3_no_hla/weights.
python ldsc/ldsc.py --h2-cts /home/eramamur/resources/gwas/Kunkle_Stage1.sumstats.gz --ref-ld-chr /home/eramamur/resources/1000G_EUR_Phase3_baseline/baseline. --out kunkle_enrichments --ref-ld-chr-cts enrichment.ldcts --w-ld-chr /home/eramamur/resources/weights_hm3_no_hla/weights. 
