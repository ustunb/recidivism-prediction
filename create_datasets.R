################################################################################
#
# Code for Interpretable Models for Recidivism Prediction
# by Jiaming Zeng/Berk Ustun/Cynthia Rudin
#
# Contact: jiaming@alum.mit.edu / ustunb@mit.edu
#
################################################################################
#
#This script will create the following data files that are used to fit models:
#
#- arrest.RData / arrest.mat
#- drug.RData / drug.mat
#- general_violence.RData / general_violence.mat
#- domestic_violence.RData / domestic_violence.mat
#- sexual_violence.RData / sexual_violence.mat
#- fatal_violence.RData / fatal_violence.mat
#- white.RData / white.mat
#- black.RData / black.mat
#- hispanic.RData / hispanic.mat
#
#To use this script
#
#1. Make sure the current working directory contains the following files:
#   - test_indices.csv
#   - validation_indices.csv
#
#2. Change "ICPSR_dir" to the directory where you unzipped the raw data from
#   ICPSR 03355. This must contain the file:
#   - 03355-0001-Data-REST.RData

################################################################################

ICPSR_dir = "" #FILL THIS OUT

data_dir = paste0(getwd(), "/data/");
ICPSR_dir = ifelse(substr(ICPSR_dir, nchar(ICPSR_dir), nchar(ICPSR_dir)) == "/", ICPSR_dir, paste0(ICPSR_dir,"/"))
raw_data_file = paste0(ICPSR_dir, "03355-0001-Data-REST.RData");
test_indices_file = paste0(data_dir, "test_indices.csv");
validation_indices_file = paste0(data_dir, "validation_indices.csv");

#check that required files exist
stopifnot(file.exists(raw_data_file));
stopifnot(file.exists(test_indices_file));
stopifnot(file.exists(validation_indices_file));

#load required packages
library(dplyr, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);
library(R.matlab, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE);

#load raw data from ICPSR directory
raw_data = new.env();
load(file = raw_data_file, envir = raw_data);
raw_data = raw_data$data;

#restrict to cases in BJS analysis and drop identifying information
raw_data = raw_data %>%
    filter(ANALYSIS == "CASE IN BJS ANALYSIS") %>%
    select(-CASENUM,
           -MNTHOB1,
           -DAYOB1,
           -MNTHOB2,
           -DAYOB2,
           -HAIR,
           -EYE);

#overwrite NAs
raw_data[is.na(raw_data)] = "UNKNOWN"

##### Process Features and Outcome Variables

#offense-related information
process_offense_information = function(raw_data){

    N = nrow(raw_data);
    raw_colnames = colnames(raw_data);

    #initialize offense codes
    offense_type_list = list(
        murder = c("MURDER", "UNSPECIFIED HOMICIDE", 1, 6),
        manslaughter = c("VOLUNTARY MANSLAUGHTER", "VEHICULAR MANSLAUGHTER", "NEGLIGENT MANSLAUGHTER", "UNSPECIFIED MANSLAUGHTER", 2,3,4,5),
        sexual_violence = c("KIDNAP","RAPE","OTHER SEXUAL ASSAULT", 7, 8, 9),
        general_violence =  c("ROBBERY","AGGRAVATED ASSAULT","OTHER VIOLENCE", 10, 11, 12),
        property = c("BURGLARY", "LARCENY", "MOTOR VEHICLE THEFT", "ARSON", "FRAUD-FORGERY-EMBEZZLEMENT", "STOLEN PROPERTY", "OTHER PROPERTY", 13, 14, 15, 16, 17, 18, 19),
        drug =  c("DRUG POSSESSION", "DRUG TRAFFICKING", "OTHER DRUG", 20, 21, 22),
        public_order = c("WEAPONS", "DUI", "OTHER PUBLIC ORDER", 23, 24, 25),
        other = c("OTHER", 26)
    );
    all_offense_types = names(offense_type_list);
    n_offense_types = length(all_offense_types);

    #initialize lists to count recorded events by crime type for each prisoner for before imprisonment
    prior_arrest_count = vector("list", n_offense_types);
    prior_arrest_count = lapply(FUN = function(x){return(rep(0,N))}, prior_arrest_count)
    names(prior_arrest_count) = all_offense_types;

    #also record additional information about arrest
    prior_arrest_detail_count = list(
        felony = rep(0,N),
        misdemeanor = rep(0,N),
        local_ordinance = rep(0,N),
        domestic_violence = rep(0,N),
        firearms = rep(0,N),
        child_victims = rep(0,N)
    );

    #initialize other lists
    prior_convict_count = prior_arrest_count;
    prior_confine_count = prior_arrest_count;
    prior_prison_count = prior_arrest_count;
    prior_probation_fine_count = prior_arrest_count;
    prior_jail_count = prior_arrest_count;
    post_arrest_count = prior_arrest_count;
    post_arrest_detail_count = prior_arrest_detail_count;

    #identify column indices for different kinds of offense-related events
    offense_ind = grep("A0[[:digit:]][[:digit:]]OFF[[:digit:]]", raw_colnames);
    offense_j_ind = grep("J0[[:digit:]][[:digit:]]OFF[[:digit:]]", raw_colnames);
    offense_convict_ind = grep("J0[[:digit:]][[:digit:]]CNV[[:digit:]]", raw_colnames);
    offense_confine_ind = grep("J0[[:digit:]][[:digit:]]CNF[[:digit:]]", raw_colnames);
    offense_jp_ind = grep("J0[[:digit:]][[:digit:]]PJP[[:digit:]]", raw_colnames);
    offense_fm_ind = grep("A0[[:digit:]][[:digit:]]FM[[:digit:]]", raw_colnames);
    offense_dmv_ind = grep("A0[[:digit:]][[:digit:]]DMV[[:digit:]]", raw_colnames);
    offense_fir_ind = grep("A0[[:digit:]][[:digit:]]FIR[[:digit:]]", raw_colnames);
    offense_cdv_ind = grep("A0[[:digit:]][[:digit:]]CDV[[:digit:]]", raw_colnames);
    year_ind = grep("A0[[:digit:]][[:digit:]]YR", raw_colnames);

    #identify the waves that mark the beginning / end of each prison sentence
    raw_data = raw_data %>%
        mutate(prisoner_id = row_number(),
               before_release_first_wave_ind = 1,
               before_release_last_wave_ind = 3*(PRIR + 1),
               after_release_first_wave_ind = before_release_last_wave_ind + 1,
               after_release_last_wave_ind = after_release_first_wave_ind + (REARR*3) - 1)

    # convert data to matrix
    offenses = as.matrix(raw_data[, offense_ind])
    offenses_j = as.matrix(raw_data[, offense_j_ind])
    offenses_convict = as.matrix(raw_data[, offense_convict_ind])
    offenses_confine = as.matrix(raw_data[, offense_confine_ind])
    offenses_jp = as.matrix(raw_data[, offense_jp_ind])
    offenses_fm = as.matrix(raw_data[, offense_fm_ind])
    offenses_dmv = as.matrix(raw_data[, offense_dmv_ind])
    offenses_fir = as.matrix(raw_data[, offense_fir_ind])
    offenses_cdv = as.matrix(raw_data[, offense_cdv_ind])

    year = as.matrix(raw_data[, year_ind])
    year = year[, rep(1:99, each = 3)];
    age_first_confinement = rep(0, N);

    for (i in 1:N) {

        # extract prior offense information
        prior_period_indices = raw_data[i, "before_release_first_wave_ind"]:raw_data[i, "before_release_last_wave_ind"];
        prior_period_indices = prior_period_indices[year[i, prior_period_indices] <= 1994];

        #get counts for each offense before prison
        prior_offenses = offenses[i, prior_period_indices];

        #prior arrest count
        prior_arrest_count$manslaughter[i] = sum(prior_offenses %in% offense_type_list$manslaughter);
        prior_arrest_count$murder[i] = sum(prior_offenses %in% offense_type_list$murder);
        prior_arrest_count$sexual_violence[i] = sum(prior_offenses %in% offense_type_list$sexual_violence);
        prior_arrest_count$general_violence[i] = sum(prior_offenses %in% offense_type_list$general_violence);
        prior_arrest_count$property[i] = sum(prior_offenses %in% offense_type_list$property);
        prior_arrest_count$drug[i] = sum(prior_offenses  %in% offense_type_list$drug);
        prior_arrest_count$public_order[i] = sum(prior_offenses %in% offense_type_list$public_order);
        prior_arrest_count$other[i] = sum(prior_offenses %in%  offense_type_list$other);

        #prior arrest details
        prior_arrest_detail_count$felony[i] = sum(offenses_fm[i, prior_period_indices] %in% c("FELONY", 1));
        prior_arrest_detail_count$misdemeanor[i] = sum(offenses_fm[i, prior_period_indices] %in% c("MISDEMEANOR", 2));
        prior_arrest_detail_count$local_ordinance[i] = sum(offenses_fm[i, prior_period_indices] %in% c("LOCAL ORDINANCE", 3));
        prior_arrest_detail_count$domestic_violence[i] = sum(offenses_dmv[i, prior_period_indices] %in% c("DOMESTIC VIOLENCE INVOLVED", 1));
        prior_arrest_detail_count$firearms[i] = sum(offenses_fir[i, prior_period_indices] %in% c("FIREARMS INVOLVED", 1));
        prior_arrest_detail_count$child_victims[i] = sum(!(offenses_cdv[i, prior_period_indices] %in% c("UNKNOWN", "NOT APPLICABLE", 98, 99)));

        #judge cycle
        convict = offenses_convict[i, prior_period_indices] == "CONVICTED";
        confine = offenses_confine[i, prior_period_indices] %in% c("CONFINED", 1);
        jail = offenses_jp[i, prior_period_indices] %in% c("JAIL", 2);
        prison = offenses_jp[i, prior_period_indices] %in% c("PRISON", 1);
        probation_fine = offenses_jp[i, prior_period_indices] %in% c("PROBATION-FINE-OTHER", 3);

        for (o in 1:n_offense_types) {
            of = offenses_j[i, prior_period_indices] %in% (offense_type_list[[o]]);
            prior_convict_count[[o]][[i]] = sum(((of + convict) == 2));
            prior_confine_count[[o]][[i]] = sum(((of + confine) == 2));
            prior_jail_count[[o]][[i]] = sum(((of + jail) == 2));
            prior_prison_count[[o]][[i]] = sum(((of + prison) == 2));
            prior_probation_fine_count[[o]][[i]] = sum(((of + probation_fine) == 2));
        }

        ## calculate age of first confinement
        idx = ifelse(any(confine), match(TRUE, confine), 0);
        idx = floor(idx/3);
        start_idx = which(colnames(raw_data) == "J001YR");
        year_of_first_confinement = raw_data[i, start_idx + idx*64];
        age_first_confinement[i] = year_of_first_confinement - raw_data[i,"YEAROB2"];

        # Offenses After Release
        after_period_indices = raw_data[i, "after_release_first_wave_ind"]:raw_data[i, "after_release_last_wave_ind"];
        has_event_after_release = (length(after_period_indices) > 0) && all(after_period_indices <= 297);
        if (has_event_after_release){
            after_period_indices = after_period_indices[year[i, after_period_indices] >= 1994];
        }
        has_event_after_release = (length(after_period_indices) > 0) && all(after_period_indices <= 297);


        if (has_event_after_release) {
            after_offenses = offenses[i, after_period_indices];

            #post arrest count
            post_arrest_count$murder[i] = sum(after_offenses %in% offense_type_list$murder);
            post_arrest_count$manslaughter[i] = sum(after_offenses %in% offense_type_list$manslaughter);
            post_arrest_count$sexual_violence[i] = sum(after_offenses %in% offense_type_list$sexual_violence);
            post_arrest_count$general_violence[i] = sum(after_offenses %in% offense_type_list$general_violence);
            post_arrest_count$property[i] = sum(after_offenses %in% offense_type_list$property);
            post_arrest_count$drug[i] = sum(after_offenses  %in% offense_type_list$drug);
            post_arrest_count$public_order[i] = sum(after_offenses %in% offense_type_list$public_order);
            post_arrest_count$other[i] = sum(after_offenses %in%  offense_type_list$other);

            #post arrest details
            post_arrest_detail_count$local_ordinance[i] = sum(offenses_fm[i, after_period_indices] %in% c("LOCAL ORDINANCE", 3));
            post_arrest_detail_count$domestic_violence[i] = sum(offenses_dmv[i, after_period_indices] %in% c("DOMESTIC VIOLENCE INVOLVED", 1));
            post_arrest_detail_count$firearms[i] = sum(offenses_fir[i, after_period_indices] %in% c("FIREARMS INVOLVED", 1));
            post_arrest_detail_count$child_victims[i] = sum(!(offenses_cdv[i, after_period_indices] %in% c("UNKNOWN", "NOT APPLICABLE", 98, 99)));
        }
    }

    offense_df = data.frame(
        #arrest history
        prior_arrests_for_felony = prior_arrest_detail_count$felony > 0,
        prior_arrests_for_misdemeanor = prior_arrest_detail_count$misdemeanor > 0,
        prior_arrests_for_local_ordinance = prior_arrest_detail_count$local_ordinance > 0,
        prior_arrests_with_firearms_involved = prior_arrest_detail_count$firearms > 0,
        prior_arrests_with_child_involved = prior_arrest_detail_count$child_victims > 0,
        prior_arrests_for_domestic_violence = prior_arrest_detail_count$domestic_violence > 0,
        prior_arrests_for_public_order = prior_arrest_count$public_order > 0,
        prior_arrests_for_drug = prior_arrest_count$drug > 0,
        prior_arrests_for_property = prior_arrest_count$property > 0,
        prior_arrests_for_sexual_violence = prior_arrest_count$sexual_violence > 0,
        prior_arrests_for_fatal_violence = (prior_arrest_count$murder + prior_arrest_count$manslaughter) > 0,
        prior_arrests_for_general_violence = prior_arrest_count$general_violence > 0,
        prior_arrests_for_multiple_types_of_crime = ((prior_arrest_detail_count$local_ordinance > 0) +
                                                         (prior_arrest_detail_count$domestic_violence > 0) +
                                                         (prior_arrest_count$public_order > 0) +
                                                         (prior_arrest_count$drug > 0) +
                                                         (prior_arrest_count$property > 0) +
                                                         (prior_arrest_count$murder > 0) +
                                                         (prior_arrest_count$manslaughter > 0) +
                                                         (prior_arrest_count$general_violence > 0) +
                                                         (prior_arrest_count$sexual_violence > 0)) > 1,
        #
        #conviction-related
        age_first_confinement = age_first_confinement,
        any_prior_jail_time = apply(data.frame(prior_jail_count), MARGIN  = 1, FUN = function(x){return(sum(x)>0)}),
        multiple_prior_jail_time = apply(data.frame(prior_jail_count), MARGIN  = 1, FUN = function(x){return(sum(x)>1)}),
        multiple_prior_prison_time = apply(data.frame(prior_prison_count), MARGIN  = 1, FUN = function(x){return(sum(x)>1)}),
        any_prior_prb_or_fine = apply(data.frame(prior_probation_fine_count), MARGIN  = 1, FUN = function(x){return(sum(x)>0)}),
        multiple_prior_prb_or_fine = apply(data.frame(prior_probation_fine_count), MARGIN  = 1, FUN = function(x){return(sum(x)>1)}),
        #
        #offenses after release from jail (outcome variables)
        arrest = raw_data$REARRD == "REARRESTED",
        domestic_violence = post_arrest_detail_count$domestic_violence > 0,
        drug = post_arrest_count$drug > 0,
        fatal_violence = (post_arrest_count$murder + post_arrest_count$manslaughter) > 0,
        sexual_violence = post_arrest_count$sexual_violence > 0,
        general_violence = post_arrest_count$general_violence > 0
    )

    return(offense_df);
}
offense_df = process_offense_information(raw_data); #warning: this step may take a while;
raw_data$age_first_confinement  = as.numeric(offense_df$age_first_confinement);

#prisoner-related information
data = raw_data %>%
    rowwise() %>%
    mutate(
        #
        time_served = TMSRVC,
        YEAROB = ifelse(YEAROB2 == 9999, YEAROB1, YEAROB2),
        ageAD = min(YEARAD - YEAROB, RELAGE),
        age_first_arrest = floor(min(A001YR - YEAROB, ageAD)),
        age_first_confinement = floor(min(age_first_confinement, ageAD)),
        age_at_release = floor(RELAGE),
        #
        female = SEX == "FEMALE",
        prior_alcohol_abuse = ALCABUS == "INMATE IS AN ALCOHOL ABUSER",
        prior_drug_abuse = DRUGAB == "INMATE IS A DRUG ABUSER",
        #
        age_at_release_leq_17 = between(age_at_release, 14, 17),
        age_at_release_18_to_24 = between(age_at_release, 18, 24),
        age_at_release_25_to_29 = between(age_at_release, 25, 29),
        age_at_release_30_to_39 = between(age_at_release, 30, 39),
        age_at_release_geq_40 = between(age_at_release, 40, 100),
        #
        age_first_arrest_leq_17 = between(age_first_arrest, 14, 17),
        age_first_arrest_18_24 = between(age_first_arrest, 18, 24),
        age_first_arrest_25_29 = between(age_first_arrest, 25, 29),
        age_first_arrest_30_39 = between(age_first_arrest, 30, 39),
        age_first_arrest_geq_40 = between(age_first_arrest, 40, 100),
        #
        age_first_confinement_leq_17 = between(age_first_confinement, 14, 17),
        age_first_confinement_18_to_24 = between(age_first_confinement, 18, 24),
        age_first_confinement_25_to_29 = between(age_first_confinement, 25, 29),
        age_first_confinement_30_to_39 = between(age_first_confinement, 30, 39),
        age_first_confinement_geq_40 = between(age_first_confinement, 40, 100),
        #
        infraction_in_prison = NFRCTNS == "INMATE HAS RECORD",
        time_served_leq_6mo    = time_served == "1 TO 6 MONTHS",
        time_served_7_to_12mo  = time_served == "7 TO 12 MONTHS",
        time_served_13_to_24mo = time_served %in% c("13 TO 18 MONTHS", "19 TO 24 MONTHS"),
        time_served_25_to_60mo = time_served %in% c("25 TO 30 MONTHS", "31 TO 36 MONTHS", "37 TO 60 MONTHS"),
        time_served_geq_61mo   = time_served == "61 MONTHS AND HIGHER",
        released_unconditonal = RELTYP %in% c("EXPIRATION OF SENTENCE", "COMMUTATION-PARDON","OTHER UNCONDITIONAL RELEASE"),
        released_conditional = RELTYP %in% c("PAROLE BOARD DECISION-SERVED NO MINIMUM","MANDATORY PAROLE RELEASE",  "PROBATION RELEASE-SHOCK PROBATION", "OTHER CONDITIONAL RELEASE"),
        #
        #
        no_prior_arrests = PRIRCAT == "1 PRIOR ARREST",
        prior_arrests_geq_1 = PRIRCAT %in% c("2 PRIOR ARRESTS","3 PRIOR ARRESTS","4 PRIOR ARRESTS","5 PRIOR ARRESTS","6 PRIOR ARRESTS","7 TO 10 PRIOR ARRESTS","11 TO 15 PRIOR ARRESTS","16 TO HI PRIOR ARRESTS"),
        prior_arrests_geq_2 = PRIRCAT %in% c("3 PRIOR ARRESTS","4 PRIOR ARRESTS","5 PRIOR ARRESTS","6 PRIOR ARRESTS","7 TO 10 PRIOR ARRESTS","11 TO 15 PRIOR ARRESTS","16 TO HI PRIOR ARRESTS"),
        prior_arrests_geq_5 = PRIRCAT %in% c("5 PRIOR ARRESTS","6 PRIOR ARRESTS","7 TO 10 PRIOR ARRESTS","11 TO 15 PRIOR ARRESTS","16 TO HI PRIOR ARRESTS"),
        #
        white = RACE == "WHITE",
        black = RACE == "BLACK",
        hispanic = ETHNIC == "HISPANIC"
    ) %>%
    ungroup() %>%
    select(female,
           prior_alcohol_abuse,
           prior_drug_abuse,
           starts_with("age_at_release_"),
           starts_with("age_first_arrest_"),
           starts_with("age_first_confinement_"),
           no_prior_arrests,
           starts_with("prior_arrests_"),
           starts_with("time_served_"),
           infraction_in_prison,
           released_unconditonal,
           released_conditional,
           white,
           black,
           hispanic)


##### Save Data Files to Fit Models in R and MATLAB ####

#feature matrix
X_all = bind_cols(data, offense_df) %>%
    select(female,
           prior_drug_abuse,
           prior_alcohol_abuse,
           starts_with("time_served_"),
           starts_with("age_at_release_"),
           starts_with("age_first_arrest_"),
           starts_with("age_first_confinement_"),
           infraction_in_prison,
           no_prior_arrests,
           starts_with("prior_arrests_geq"),
           starts_with("released_"),
           prior_arrests_for_felony,
           prior_arrests_for_misdemeanor,
           prior_arrests_for_local_ordinance,
           prior_arrests_for_domestic_violence,
           prior_arrests_for_public_order,
           prior_arrests_for_drug,
           prior_arrests_for_property,
           prior_arrests_for_sexual_violence,
           prior_arrests_for_fatal_violence,
           prior_arrests_for_general_violence,
           prior_arrests_with_firearms_involved,
           prior_arrests_with_child_involved,
           prior_arrests_for_multiple_types_of_crime,
           any_prior_jail_time,
           multiple_prior_jail_time,
           multiple_prior_prison_time,
           any_prior_prb_or_fine,
           multiple_prior_prb_or_fine);

#outcomes
Y_all = bind_cols(
    # offenses
    offense_df %>% select(arrest,
                          drug,
                          domestic_violence,
                          fatal_violence,
                          sexual_violence,
                          general_violence),
    # race
    data %>% select(white,
                    black,
                    hispanic)
)

#force all booleans to numeric values
X_all = X_all + 0;
Y_all = Y_all + 0;
X_all = as.matrix(X_all);
Y_all = as.matrix(Y_all);

#load test indices
test_indices = read.csv(file = test_indices_file, header = FALSE);
test_indices = test_indices == 1

#load validation indices
folds = read.csv(file = validation_indices_file, header = FALSE);

#save R files for each outcome
for (outcome in colnames(Y_all)){
    X_test = X_all[test_indices,]
    Y_test = Y_all[test_indices, outcome];
    X = X_all[!test_indices,]
    Y = Y_all[!test_indices, outcome]
    save(X, Y, X_test, Y_test, folds, file = paste0(data_dir, outcome, ".RData"));
}

#save MATLAB files for each outcome
for (outcome in colnames(Y_all)){

    X = cbind(1.0, X_all[!test_indices,]);
    X_test = cbind(1.0, X_all[test_indices,]);
    X_names = c("(Intercept)", colnames(X_all))
    colnames(X) = X_names;
    colnames(X_test) = X_names;

    Y = as.matrix(Y_all[!test_indices, outcome])
    Y_test = as.matrix(Y_all[test_indices, outcome]);
    Y[Y<=0,] = -1;
    Y_test[Y_test<=0,] = -1;

    writeMat(con = paste0(data_dir, outcome, ".mat"),
             X_test = X_test,
             Y_test = Y_test,
             X = X,
             Y = Y,
             X_names = X_names,
             Y_name = outcome,
             folds = folds$V1);
}