import os

def generate_pmf_ini(run_dir, n_rows, n_cols, factors, fpeak, error_fraction, task_name="PyNSD"):
    ini_path = os.path.join(run_dir, f"{task_name}.ini")                         # Target path

    lines = [
        f" ##PMF2 .ini file for: analysis of {task_name} Data",
        " ## Monitor code M: if M>1, PMF2 writes output every Mth step",
        " ## For finding errors, use M<1 to output debug information",
        " ##      M       PMF2 version number",
        "         2          4.2",
        " ## Dimensions: Rows, Columns, Factors. Number of \"Repeats\"",
        f"             {n_rows}       {n_cols}       {factors}         1",     # Dynamic dims
        " ##   \"FPEAK\"  (>0.0 for large values and zeroes on F side)",
        f" {fpeak:.4f}",                                                         # Dynamic FPEAK
        " ## Mode(T:robusTRUE, F:non-robust)  Outlier-distance        (T=True F=False)",
        "               T                         4.000",
        " ## Codes C1 C2 C3 for X_std-dev, Errormodel EM=[-10 ... -14]",
        "         0.000    0.0000    0.0000     -12",                             # Python handles C3
        " ## G Background fit:  Components   Pullup_strength",
        "                              0       0.0000",
        " ## Pseudorandom numbers:  Seed     Initially skipped",
        "                              3          0",
        " ## Iteration control table for 3 levels of limit repulsion \"lims\"",
        " ##  \"lims\"    Chi2_test  Ministeps_required  Max_cumul_count",
        "      10.00000   0.50000            5           400",
        "       0.03000   0.50000            5           600",
        "       0.00010   0.30000            5           800",
        " ## Table of FORMATs, with reference numbers from 50 to 59",
        " ## Number  Format_text(max 40 chars)",
        "       50   \"(A)                                     \"",
        "       51   \"((1X,5G13.5E2))                         \"",
        "       52   \"((1X,10F8.3))                           \"",
        "       53   \"((1X,20(I3,:' ')))                      \"",
        "       54   \"((1X,150(G12.5E1,:' ')))                \"",
        "       55   \"((1X,180(F9.4,:' ')))                   \"",
        "       56   \"(1X,A)                                  \"",
        "       57   \"((1X,370(G13.5E2,:',')))                \"",
        "       58   \"((1X,350(F4.3,:' ')))                   \"",
        "       59   \"((1X,600(I2,:' ')))                     \"",
        " ## Table of file properties, with reference numbers from 30 to 39",
        " ## Num- In  Opening  Max-rec File-name(max 40 chars)",
        " ## ber  T/F status   length",
        "     30   T \"OLD    \" 32000 \"MATRIX.DAT                              \"", # Aligned 32000
        "     31   T \"OLD    \" 32000 \"T_MATRIX.DAT                            \"", # Aligned 32000
        "     32   F \"OLD    \" 32000 \"U_MATRIX.DAT                            \"", # Aligned 32000
        "     33   T \"OLD    \" 32000 \"V_MATRIX.DAT                            \"", # Aligned 32000
        "     34   F \"REPLACE\" 32000 \"ScaledResid.dat                         \"", # Aligned 32000
        "     35   F \"UNKNOWN\" 32000 \"MISC.TXT                                \"", # Aligned 32000
        "     36   F \"REPLACE\" 32000 \"G_FACTOR.TXT                            \"", # Aligned 32000
        "     37   F \"REPLACE\" 32000 \"F_FACTOR.TXT                            \"", # Aligned 32000
        "     38   F \"REPLACE\" 32000 \"TEMP.TXT                                \"", # Aligned 32000
        "     39   T \"OLD    \" 32000 \"FKEY.DAT                                \"", # Aligned 32000
        " ## Input/output definitions for 21 matrices",
        " ##  ===HEADING=====   ========MATRIX==========       default HEADING",
        " ##  --IN---- --OUT-   -----IN------   ---OUT--       for each matrix",
        " ## FIL(R)FMT FIL FMT FIL(R)(C)FMT(T) FIL FMT(T) ------max 40 chars----...",
        "      0 F  50  38  56  30 F      0 F   38  57 F  \"X (data matr)          \"",
        "      0 F  50  38  56  31 F      0 F   38  57 F  \"X_std-dev /T (constant)\"",
        "      0 F  50   0  56   0 F      0 F    0  57 F  \"X_std-dev /U (sqrt)    \"",
        "      0 F  50   0  56  33 F      0 F    0  57 F  \"X_std-dev /V (proport) \"",
        "      0 F  50   0  56   0 F  F   0 F    0  57 F  \"Factor G(orig.)        \"",
        "      0 F  50  38  56   0 F  F   0 F   38  57 F  \"Factor F(orig.)        \"",
        "      0 F  50   0  56   0 F      0 F    0  53 F  \"Key (factor G)         \"",
        "      0 F  50   0  56   0 F      0 F    0  59 F  \"Key (factor F)         \"",
        "      0 F  50   0  56   0 F      0 F    0  52 F  \"Rotation commands      \"",
        "      0 F  50   0  56                  36  57 F  \"Computed Factor G Q=   \"",
        "      0 F  50   0  56                  37  57 F  \"Computed Factor F      \"",
        "      0 F  50  35  56                  35  57 F  \"Computed std-dev of G  \"",
        "      0 F  50  35  56                  35  57 F  \"Computed std-dev of F  \"",
        "      0 F  50   0  56                   0  57 F  \"G_explained_variation  \"",
        "      0 F  50  35  56                  35  58 F  \"F_explained_variation  \"",
        "      0 F  50  35  56                  35  57 F  \"Residual matrix X-GF   \"",
        "      0 F  50  35  56                  34  57 F  \"Scaled resid. (X-GF)/S \"",
        "      0 F  50   0  56                   0  57 F  \"Robustized residual    \"",
        "      0 F  50  35  56                  35  55 F  \"Rotation estimates.  Q=\"",
        "      0 F  50  38  56                  38  57 F  \"Computed X_std-dev     \"",
        "      0 F  50   0  56                   0  55 F  \"Background coefficients\"",
        " ## If Repeats>1, for input matrices, select (R)=T or (C)=T or none",
        " ##    (R)=T: read(generate) again   (C)=TRUE,\"chain\": use computed G or F",
        " ##    none, i.e.(R)=FALSE,(C)=F: use same value as in first task",
        " ## (T)=T: Matrix should be read/written in Transposed shape",
        " ##",
        " ## Normalization of factor vectors before output. Select one of:",
        " ##   None   MaxG=1   Sum|G|=1 Mean|G|=1  MaxF=1 Sum|F|=1 Mean|F|=1",
        "         F        F        F        T        F        F        F",
        " ## Special/read layout for X (and for X_std-dev on following line)",
        " ## Values-to-read (0: no special) #-of-X11  incr-to-X12  incr-to-X21",
        "                             0         0         0         0",
        "                             0         0         0         0",
        " ## A priori linear constraints for factors, file name: (not yet available)",
        "     \"none                                      \"",
        " ## Optional parameter lines (insert more lines if needed)",
        " missingneg 10",
        "",                                                                        # Stop parameters
        " ## (FIL#4 = this file)    (FIL#24 = .log file)",
        " ## After next 2 lines, you may include matrices to be read with FIL=4",
        " ## but observe maximum line length = 120 characters in this file",
        " ## and maximum line length = 255 characters in the .log file",
        "",                                                                        # Empty Line 1 for Group 17
        "",                                                                        # Empty Line 2 for Group 17
        " ",                                                                       # Mandatory final whitespace
        " "                                                                        # Safety buffer
    ]

    with open(ini_path, 'w') as f:
        f.write("\n".join(lines))