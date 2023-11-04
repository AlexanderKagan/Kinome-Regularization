library(dplyr)
library(vbvs.concurrent)
library(refund.shiny)

train_test_split <- function(data, test_size, seed){
  drug_names <- data$subj
  drug_names_matrix <- matrix(drug_names)
  unique_drugs <- unique(drug_names_matrix)
  num_to_select <- round(test_size * length(unique_drugs))
  set.seed(seed)
  test_index <- sample(seq_along(unique_drugs), num_to_select)
  train_index <- setdiff(seq_along(unique_drugs), test_index)
  return (list(test_index, train_index))
}

CV_regressor <- function(base_path, data, train_index, test_index, v0, Kp, Kt, A, B) {
  data_test <- data[test_index, ]
  data_train <- data[train_index, ]

  base_path = paste(base_path, "v0_", v0, "_Kp_", Kp, "_Kt_", Kt, "_A_", A, "_B_", B, "/", sep = "")
  dir.create(base_path, recursive = TRUE)
  model <- Y ~ Cov_1 + Cov_2 + Cov_3 + Cov_4 + Cov_5 + Cov_6 + Cov_7 + Cov_8 + Cov_9 + Cov_10 + Cov_11 + Cov_12 + Cov_13 + Cov_14 + Cov_15 + Cov_16 + Cov_17 + Cov_18 + Cov_19 + Cov_20 + Cov_21 + Cov_22 + Cov_23 + Cov_24 + Cov_25 + Cov_26 + Cov_27 + Cov_28 + Cov_29 + Cov_30 + Cov_31 + Cov_32 + Cov_33 + Cov_34 + Cov_35 + Cov_36 + Cov_37 + Cov_38 + Cov_39 + Cov_40 + Cov_41 + Cov_42 + Cov_43 + Cov_44 + Cov_45 + Cov_46 + Cov_47 + Cov_48 + Cov_49 + Cov_50 + Cov_51 + Cov_52 + Cov_53 + Cov_54 + Cov_55 + Cov_56 + Cov_57 + Cov_58 + Cov_59 + Cov_60 + Cov_61 + Cov_62 + Cov_63 + Cov_64 + Cov_65 + Cov_66 + Cov_67 + Cov_68 + Cov_69 + Cov_70 + Cov_71 + Cov_72 + Cov_73 + Cov_74 + Cov_75 + Cov_76 + Cov_77 + Cov_78 + Cov_79 + Cov_80 + Cov_81 + Cov_82 + Cov_83 + Cov_84 + Cov_85 + Cov_86 + Cov_87 + Cov_88 + Cov_89 + Cov_90 + Cov_91 + Cov_92 + Cov_93 + Cov_94 + Cov_95 + Cov_96 + Cov_97 + Cov_98 + Cov_99 + Cov_100 + Cov_101 + Cov_102 + Cov_103 + Cov_104 + Cov_105 + Cov_106 + Cov_107 + Cov_108 + Cov_109 + Cov_110 + Cov_111 + Cov_112 + Cov_113 + Cov_114 + Cov_115 + Cov_116 + Cov_117 + Cov_118 + Cov_119 + Cov_120 + Cov_121 + Cov_122 + Cov_123 + Cov_124 + Cov_125 + Cov_126 + Cov_127 + Cov_128 + Cov_129 + Cov_130 + Cov_131 + Cov_132 + Cov_133 + Cov_134 + Cov_135 + Cov_136 + Cov_137 + Cov_138 + Cov_139 + Cov_140 + Cov_141 + Cov_142 + Cov_143 + Cov_144 + Cov_145 + Cov_146 + Cov_147 + Cov_148 + Cov_149 + Cov_150 + Cov_151 + Cov_152 + Cov_153 + Cov_154 + Cov_155 + Cov_156 + Cov_157 + Cov_158 + Cov_159 + Cov_160 + Cov_161 + Cov_162 + Cov_163 + Cov_164 + Cov_165 + Cov_166 + Cov_167 + Cov_168 + Cov_169 + Cov_170 + Cov_171 + Cov_172 + Cov_173 + Cov_174 + Cov_175 + Cov_176 + Cov_177 + Cov_178 + Cov_179 + Cov_180 + Cov_181 + Cov_182 + Cov_183 + Cov_184 + Cov_185 + Cov_186 + Cov_187 + Cov_188 + Cov_189 + Cov_190 + Cov_191 + Cov_192 + Cov_193 + Cov_194 + Cov_195 + Cov_196 + Cov_197 + Cov_198 + Cov_199 + Cov_200 + Cov_201 + Cov_202 + Cov_203 + Cov_204 + Cov_205 + Cov_206 + Cov_207 + Cov_208 + Cov_209 + Cov_210 + Cov_211 + Cov_212 + Cov_213 + Cov_214 + Cov_215 + Cov_216 + Cov_217 + Cov_218 + Cov_219 + Cov_220 + Cov_221 + Cov_222 + Cov_223 + Cov_224 + Cov_225 + Cov_226 + Cov_227 + Cov_228 + Cov_229 + Cov_230 + Cov_231 + Cov_232 + Cov_233 + Cov_234 + Cov_235 + Cov_236 + Cov_237 + Cov_238 + Cov_239 + Cov_240 + Cov_241 + Cov_242 + Cov_243 + Cov_244 + Cov_245 + Cov_246 + Cov_247 + Cov_248 + Cov_249 + Cov_250 + Cov_251 + Cov_252 + Cov_253 + Cov_254 + Cov_255 + Cov_256 + Cov_257 + Cov_258 + Cov_259 + Cov_260 + Cov_261 + Cov_262 + Cov_263 + Cov_264 + Cov_265 + Cov_266 + Cov_267 + Cov_268 + Cov_269 + Cov_270 + Cov_271 + Cov_272 + Cov_273 + Cov_274 + Cov_275 + Cov_276 + Cov_277 + Cov_278 + Cov_279 + Cov_280 + Cov_281 + Cov_282 + Cov_283 + Cov_284 + Cov_285 + Cov_286 + Cov_287 + Cov_288 + Cov_289 + Cov_290 + Cov_291 + Cov_292 + Cov_293 + Cov_294 + Cov_295 + Cov_296 + Cov_297 + Cov_298 + Cov_299 + Cov_300 + Cov_301 + Cov_302 + Cov_303 + Cov_304 + Cov_305 + Cov_306 + Cov_307 + Cov_308 + Cov_309 + Cov_310 + Cov_311 + Cov_312 + Cov_313 + Cov_314 + Cov_315 + Cov_316 + Cov_317 + Cov_318 + Cov_319 + Cov_320 + Cov_321 + Cov_322 + Cov_323 + Cov_324 + Cov_325 + Cov_326 + Cov_327 + Cov_328 + Cov_329 + Cov_330 + Cov_331 + Cov_332 + Cov_333 + Cov_334 + Cov_335 + Cov_336 + Cov_337 + Cov_338 + Cov_339 + Cov_340 + Cov_341 + Cov_342 + Cov_343 + Cov_344 + Cov_345 + Cov_346 + Cov_347 + Cov_348 + Cov_349 + Cov_350 + Cov_351 + Cov_352 + Cov_353 + Cov_354 + Cov_355 + Cov_356 + Cov_357 + Cov_358 + Cov_359 + Cov_360 + Cov_361 + Cov_362 + Cov_363 + Cov_364 + Cov_365 + Cov_366 + Cov_367 + Cov_368 + Cov_369 | time

  fit_vbvs = vbvs_concurrent(model, id.var = "subj", Kp = Kp, Kt = Kt, data = data_train, t.min = 0.0, t.max = 10.0, standardized = FALSE, v0 = v0, A = A, B = B, Atheta = A, Btheta = B)

  betas_path <- paste(base_path, "train_beta.csv", sep = "")
  write.csv(fit_vbvs$beta.pm, betas_path, row.names=FALSE)

  y_train_predicted <- fit_vbvs$Yhat
  y_train_predicted_path <- paste(base_path, "y_train_predicted.csv", sep = "")
  write.csv(y_train_predicted, y_train_predicted_path, row.names=FALSE)

  y_train_true_standartized <- fit_vbvs$data.model$Y
  y_train_true_standartized_path <- paste(base_path, "y_train_true_standartized.csv", sep = "")
  write.csv(y_train_true_standartized, y_train_true_standartized_path, row.names=FALSE)

  y_test_predicted <- predict(fit_vbvs, data.new = data_test, standardized = FALSE)
  y_test_predicted_path <- paste(base_path, "y_test_predicted.csv", sep = "")
  write.csv(y_test_predicted, y_test_predicted_path, row.names=FALSE)

  y_test_true_standartized <- standardize_variables(data.original = data_train, data.new = data_test, time.var = "time", LHS = "Y", trmstrings = c("Cov_1"))$Y
  y_test_true_standartized_path <- paste(base_path, "y_test_true_standartized.csv", sep = "")
  write.csv(y_test_true_standartized, y_test_true_standartized_path, row.names=FALSE)

  gamma_path <- paste(base_path, "gamma.csv", sep = "")
  write.csv(fit_vbvs$gamma.pm, gamma_path, row.names=FALSE)

  return (list(y_train_predicted, y_train_true_standartized, y_test_predicted, y_test_true_standartized, fit_vbvs))
}

hyperparameters_config <- list(
  list(v0 = 1, Kp = 7, Kt = 4, A = 0.5, B = 0.5),
  list(v0 = 1, Kp = 7, Kt = 4, A = 1.0, B = 1.0)
)

data <- read.csv("./fof/data/r/prepared_data.csv")
split <- train_test_split(data, test_size=0.2, seed=42)
test_index <- split[[1]]
train_index <- split[[2]]

base_path <- "./fof/data/r/hyperparameter_tunning/"

for (hyperparameters in hyperparameters_config){
  print("ITERATION")
  print(hyperparameters)
  v0 <- hyperparameters$v0
  Kp <- hyperparameters$Kp
  Kt <- hyperparameters$Kt
  A <- hyperparameters$A
  B <- hyperparameters$B
  final_result <- CV_regressor(base_path, data, train_index, test_index, v0, Kp, Kt, A, B)
  # y_train_predicted <- final_result[[1]]
  # y_train_true_standartized <- final_result[[2]]
  # y_test_predicted <- final_result[[3]]
  # y_test_true_standartized <- final_result[[4]]
  model <- final_result[[5]]
  print('GAMMA SUM')
  print(sum(model$gamma.pm))
  print('GAMMA > 0.01 count')
  print(sum(model$gamma.pm > 0.01))
}