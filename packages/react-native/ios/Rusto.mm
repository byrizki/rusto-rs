#import "Rusto.h"
#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *text;
    float score;
    float box_x1;
    float box_y1;
    float box_x2;
    float box_y2;
    float box_x3;
    float box_y3;
    float box_x4;
    float box_y4;
    float frame_width;
    float frame_height;
    float frame_top;
    float frame_left;
} CTextResult;

typedef void *ROCRHandle;

ROCRHandle rocr_initialize(const char *config_json);
int rocr_detect_text_file(ROCRHandle handle, const char *image_path, const char *options_json, void **results, size_t *count);
char *rocr_detect_text_file_spatial(ROCRHandle handle, const char *image_path, const char *options_json);
int rocr_detect_text_data(ROCRHandle handle, const void *data, size_t length, const char *options_json, void **results, size_t *count);
char *rocr_detect_text_data_spatial(ROCRHandle handle, const void *data, size_t length, const char *options_json);
void rocr_free_string(char *s);
void rocr_free_results(void *results, size_t count);
void rocr_free(ROCRHandle handle);

#ifdef __cplusplus
}
#endif

@interface Rusto () {
    ROCRHandle _rustoInstance;
}
@end

@implementation Rusto

RCT_EXPORT_MODULE(Rusto);

+ (BOOL)requiresMainQueueSetup {
    return NO;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _rustoInstance = NULL;
    }
    return self;
}

- (void)dealloc {
    if (_rustoInstance != NULL) {
        rocr_free(_rustoInstance);
        _rustoInstance = NULL;
    }
}

static NSError *makeValidationError(NSString *message) {
    return [NSError errorWithDomain:@"@rustors/react-native"
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

- (NSString *)resolveFilePath:(NSString *)rawPath {
    NSString *path = [rawPath stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if ([path hasPrefix:@"file://"]) {
        NSURL *url = [NSURL URLWithString:path];
        if (url != nil && url.path != nil) {
            path = url.path;
        } else {
            path = [path substringFromIndex:7];
        }
    } else if ([path hasPrefix:@"file:"]) {
        path = [path substringFromIndex:5];
    }
    return [path stringByRemovingPercentEncoding] ?: path;
}

- (NSString *)copyToCache:(NSString *)sourcePath filename:(NSString *)filename {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
    if (paths.count == 0) return sourcePath;
    NSString *cacheDir = paths[0];
    NSString *cachePath = [cacheDir stringByAppendingPathComponent:filename];
    NSFileManager *fm = [NSFileManager defaultManager];
    if ([fm fileExistsAtPath:cachePath]) {
        return cachePath;
    }
    NSError *error = nil;
    if ([fm copyItemAtPath:sourcePath toPath:cachePath error:&error]) {
        return cachePath;
    }
    return sourcePath;
}

- (NSString *)getResourcePath:(NSString *)filename {
    if ([filename hasPrefix:@"/"] && [[NSFileManager defaultManager] fileExistsAtPath:filename]) {
        return filename;
    }

    NSString *sourcePath = nil;
    NSString *name = [[filename stringByReplacingOccurrencesOfString:@".rten" withString:@""]
                     stringByReplacingOccurrencesOfString:@".onnx" withString:@""];
    name = [[name stringByReplacingOccurrencesOfString:@".mnn" withString:@""]
            stringByReplacingOccurrencesOfString:@".txt" withString:@""];
    NSString *ext = [filename pathExtension];
    NSMutableArray<NSString *> *candidateExts = [NSMutableArray array];
    if (ext.length > 0) [candidateExts addObject:ext];
    for (NSString *candidate in @[@"rten", @"onnx", @"mnn", @"txt"]) {
        if (![candidateExts containsObject:candidate]) {
            [candidateExts addObject:candidate];
        }
    }

    NSArray<NSString *> *bundleNames = @[
        @"RustOModels", @"RustoModels",
        @"RustOModels_PPOCRv6_Tiny", @"RustOModels_PPOCRv6_Small", @"RustOModels_PPOCRv6_Medium",
        @"RustOModels_PPOCRv5_Mobile", @"RustOModels_PPOCRv5_Server",
        @"RustOModels_PPOCRv4_Mobile", @"RustOModels_PPOCRv4_Server"
    ];

    NSBundle *moduleBundle = [NSBundle bundleForClass:[self class]];
    for (NSString *bName in bundleNames) {
        NSString *bundlePath = [moduleBundle pathForResource:bName ofType:@"bundle"] ?:
                               [[NSBundle mainBundle] pathForResource:bName ofType:@"bundle"];
        if (bundlePath != nil) {
            NSBundle *bundle = [NSBundle bundleWithPath:bundlePath];
            if (bundle != nil) {
                for (NSString *candidateExt in candidateExts) {
                    NSString *filePath = [bundle pathForResource:name ofType:candidateExt];
                    if (filePath != nil) {
                        sourcePath = filePath;
                        break;
                    }
                }
                if (sourcePath != nil) break;
            }
        }
    }

    if (sourcePath == nil) {
        sourcePath = [[NSBundle mainBundle] pathForResource:filename ofType:nil];
    }

    if (sourcePath == nil) {
        NSArray *docPaths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
        if (docPaths.count > 0) {
            NSString *docPath = [docPaths[0] stringByAppendingPathComponent:filename];
            if ([[NSFileManager defaultManager] fileExistsAtPath:docPath]) {
                sourcePath = docPath;
            }
        }
    }

    if (sourcePath != nil) {
        return [self copyToCache:sourcePath filename:filename];
    }
    return nil;
}

- (NSDictionary *)validatedInitializeConfig:(NSDictionary *)configDict error:(NSError **)error {
    NSDictionary *config = configDict ?: @{};
    for (id value in config.allValues) {
        if ([value isKindOfClass:[NSNull class]]) {
            if (error) *error = makeValidationError(@"configuration values must not be null");
            return nil;
        }
    }

    NSSet *allowed = [NSSet setWithObjects:@"preset", @"models", nil];
    NSSet *keys = [NSSet setWithArray:config.allKeys];
    if (![keys isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"unknown key");
        return nil;
    }

    if (config[@"preset"] != nil) {
        id preset = config[@"preset"];
        if (![preset isKindOfClass:[NSString class]] ||
            ![@[@"ppv6", @"ppv5", @"ppv4", @"ppv3"] containsObject:preset]) {
            if (error) *error = makeValidationError(@"preset is invalid");
            return nil;
        }
    }

    NSMutableDictionary *validated = [config mutableCopy];
    if (config[@"models"] != nil) {
        id rawModels = config[@"models"];
        if (![rawModels isKindOfClass:[NSDictionary class]]) {
            if (error) *error = makeValidationError(@"models must be an object");
            return nil;
        }
        NSDictionary *models = (NSDictionary *)rawModels;
        NSSet *allowedModels = [NSSet setWithObjects:@"detection", @"recognition", @"dictionary", @"classification", @"orientation", nil];
        NSSet *modelKeys = [NSSet setWithArray:models.allKeys];
        if (![modelKeys isSubsetOfSet:allowedModels]) {
            if (error) *error = makeValidationError(@"models contains an unknown key");
            return nil;
        }

        NSMutableDictionary<NSString *, NSString *> *strings = [NSMutableDictionary dictionary];
        for (NSString *key in models) {
            id val = models[key];
            if (![val isKindOfClass:[NSString class]] ||
                [[val stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] length] == 0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"models.%@ must be a non-empty string", key]);
                return nil;
            }
            strings[key] = val;
        }
        validated[@"models"] = strings;
    }

    return validated;
}

- (NSDictionary *)validatedRuntimeDetection:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"detection must be an object");
        return nil;
    }
    NSDictionary *detection = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"limitSideLen", @"limitType", @"mean", @"std", nil];
    NSSet *keys = [NSSet setWithArray:detection.allKeys];
    if (![keys isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"detection contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if (detection[@"limitSideLen"] != nil) {
        id rawVal = detection[@"limitSideLen"];
        if (![rawVal isKindOfClass:[NSNumber class]] || [rawVal isKindOfClass:[@YES class]] ||
            CFGetTypeID((__bridge CFTypeRef)rawVal) == CFBooleanGetTypeID()) {
            if (error) *error = makeValidationError(@"limitSideLen is invalid");
            return nil;
        }
        double val = [rawVal doubleValue];
        if (!isfinite(val) || val <= 0 || val > 32767 || floor(val) != val) {
            if (error) *error = makeValidationError(@"limitSideLen is invalid");
            return nil;
        }
        result[@"limitSideLen"] = rawVal;
    }
    if (detection[@"limitType"] != nil) {
        id rawVal = detection[@"limitType"];
        if (![rawVal isKindOfClass:[NSString class]] || (![rawVal isEqualToString:@"min"] && ![rawVal isEqualToString:@"max"])) {
            if (error) *error = makeValidationError(@"limitType is invalid");
            return nil;
        }
        result[@"limitType"] = rawVal;
    }
    for (NSString *key in @[@"mean", @"std"]) {
        if (detection[key] != nil) {
            id rawVal = detection[key];
            if (![rawVal isKindOfClass:[NSArray class]] || [rawVal count] != 3) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ must contain three numbers", key]);
                return nil;
            }
            for (id item in rawVal) {
                if (![item isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)item) == CFBooleanGetTypeID()) {
                    if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ contains an invalid value", key]);
                    return nil;
                }
                double d = [item doubleValue];
                if (!isfinite(d) || ([key isEqualToString:@"std"] && d == 0.0)) {
                    if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ contains an invalid value", key]);
                    return nil;
                }
            }
            result[key] = rawVal;
        }
    }
    return result;
}

- (NSDictionary *)validatedRuntimePostprocess:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"postprocess must be an object");
        return nil;
    }
    NSDictionary *postprocess = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"threshold", @"boxThreshold", @"maxCandidates", @"unclipRatio", @"useDilation", nil];
    NSSet *keys = [NSSet setWithArray:postprocess.allKeys];
    if (![keys isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"postprocess contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"threshold", @"boxThreshold"]) {
        if (postprocess[key] != nil) {
            id val = postprocess[key];
            if (![val isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)val) == CFBooleanGetTypeID()) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
            double d = [val doubleValue];
            if (!isfinite(d) || d < 0.0 || d > 1.0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
            result[key] = val;
        }
    }
    if (postprocess[@"maxCandidates"] != nil) {
        id val = postprocess[@"maxCandidates"];
        if (![val isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)val) == CFBooleanGetTypeID()) {
            if (error) *error = makeValidationError(@"maxCandidates is invalid");
            return nil;
        }
        double d = [val doubleValue];
        if (!isfinite(d) || d < 1.0 || floor(d) != d) {
            if (error) *error = makeValidationError(@"maxCandidates is invalid");
            return nil;
        }
        result[@"maxCandidates"] = val;
    }
    if (postprocess[@"unclipRatio"] != nil) {
        id val = postprocess[@"unclipRatio"];
        if (![val isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)val) == CFBooleanGetTypeID()) {
            if (error) *error = makeValidationError(@"unclipRatio is invalid");
            return nil;
        }
        double d = [val doubleValue];
        if (!isfinite(d) || d <= 0.0) {
            if (error) *error = makeValidationError(@"unclipRatio is invalid");
            return nil;
        }
        result[@"unclipRatio"] = val;
    }
    if (postprocess[@"useDilation"] != nil) {
        id val = postprocess[@"useDilation"];
        if (![val isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)val) != CFBooleanGetTypeID()) {
            if (error) *error = makeValidationError(@"useDilation must be a boolean");
            return nil;
        }
        result[@"useDilation"] = val;
    }
    return result;
}

- (NSDictionary *)validatedRuntimeOptions:(NSDictionary *)input error:(NSError **)error {
    NSMutableDictionary *options = [(input ?: @{}) mutableCopy];
    for (id val in options.allValues) {
        if ([val isKindOfClass:[NSNull class]]) {
            if (error) *error = makeValidationError(@"options values must not be null");
            return nil;
        }
    }
    NSSet *allowed = [NSSet setWithObjects:@"output", @"lineYThreshold", @"wordXThreshold",
                                          @"textScore", @"classification", @"orientation",
                                          @"minHeight", @"maxSideLen", @"minSideLen",
                                          @"widthHeightRatio", @"detection", @"postprocess", nil];
    NSSet *keys = [NSSet setWithArray:options.allKeys];
    if (![keys isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"unknown key");
        return nil;
    }

    if (options[@"output"] != nil) {
        id output = options[@"output"];
        if (![output isKindOfClass:[NSString class]] ||
            (![output isEqualToString:@"lines"] && ![output isEqualToString:@"words"] && ![output isEqualToString:@"spatial"])) {
            if (error) *error = makeValidationError(@"output is invalid");
            return nil;
        }
    } else {
        options[@"output"] = @"lines";
    }

    for (NSString *key in @[@"lineYThreshold", @"wordXThreshold"]) {
        if (options[key] != nil) {
            id val = options[key];
            if (![val isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)val) == CFBooleanGetTypeID()) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
            double d = [val doubleValue];
            if (!isfinite(d) || d < 0.0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
        }
    }

    if (options[@"textScore"] != nil) {
        id val = options[@"textScore"];
        if (![val isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)val) == CFBooleanGetTypeID()) {
            if (error) *error = makeValidationError(@"textScore is invalid");
            return nil;
        }
        double d = [val doubleValue];
        if (!isfinite(d) || d < 0.0 || d > 1.0) {
            if (error) *error = makeValidationError(@"textScore is invalid");
            return nil;
        }
    }

    for (NSString *key in @[@"classification", @"orientation"]) {
        if (options[key] != nil) {
            id val = options[key];
            if (![val isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)val) != CFBooleanGetTypeID()) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ must be a boolean", key]);
                return nil;
            }
        }
    }

    NSNumber *(^checkNum)(NSString *, BOOL) = ^NSNumber *(NSString *key, BOOL allowNegOne) {
        id raw = options[key];
        if (raw == nil) return nil;
        if (![raw isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)raw) == CFBooleanGetTypeID()) {
            return nil;
        }
        double d = [raw doubleValue];
        if (!isfinite(d) || (d <= 0.0 && !(allowNegOne && d == -1.0))) {
            return nil;
        }
        return raw;
    };

    if (options[@"maxSideLen"] != nil && checkNum(@"maxSideLen", NO) == nil) {
        if (error) *error = makeValidationError(@"maxSideLen is invalid");
        return nil;
    }
    if (options[@"minSideLen"] != nil && checkNum(@"minSideLen", NO) == nil) {
        if (error) *error = makeValidationError(@"minSideLen is invalid");
        return nil;
    }
    if (options[@"minHeight"] != nil && checkNum(@"minHeight", NO) == nil) {
        if (error) *error = makeValidationError(@"minHeight is invalid");
        return nil;
    }
    if (options[@"widthHeightRatio"] != nil && checkNum(@"widthHeightRatio", YES) == nil) {
        if (error) *error = makeValidationError(@"widthHeightRatio is invalid");
        return nil;
    }

    if (options[@"minSideLen"] != nil && options[@"maxSideLen"] != nil) {
        if ([options[@"minSideLen"] doubleValue] > [options[@"maxSideLen"] doubleValue]) {
            if (error) *error = makeValidationError(@"minSideLen must be <= maxSideLen");
            return nil;
        }
    }

    if (options[@"detection"] != nil) {
        NSDictionary *det = [self validatedRuntimeDetection:options[@"detection"] error:error];
        if (det == nil) return nil;
        options[@"detection"] = det;
    }
    if (options[@"postprocess"] != nil) {
        NSDictionary *pp = [self validatedRuntimePostprocess:options[@"postprocess"] error:error];
        if (pp == nil) return nil;
        options[@"postprocess"] = pp;
    }

    return options;
}

- (NSArray<NSDictionary *> *)convertResultsToArray:(void *)resultsPtr count:(size_t)count {
    NSMutableArray<NSDictionary *> *resultArray = [NSMutableArray arrayWithCapacity:count];
    CTextResult *results = (CTextResult *)resultsPtr;

    for (size_t i = 0; i < count; i++) {
        CTextResult *res = &results[i];
        NSString *textString = res->text ? [NSString stringWithUTF8String:res->text] : @"";

        float minX = MIN(MIN(res->box_x1, res->box_x2), MIN(res->box_x3, res->box_x4));
        float maxX = MAX(MAX(res->box_x1, res->box_x2), MAX(res->box_x3, res->box_x4));
        float minY = MIN(MIN(res->box_y1, res->box_y2), MIN(res->box_y3, res->box_y4));
        float maxY = MAX(MAX(res->box_y1, res->box_y2), MAX(res->box_y3, res->box_y4));

        NSDictionary *dict = @{
            @"text": textString,
            @"score": @((double)res->score),
            @"box_points": @[
                @[@((double)res->box_x1), @((double)res->box_y1)],
                @[@((double)res->box_x2), @((double)res->box_y2)],
                @[@((double)res->box_x3), @((double)res->box_y3)],
                @[@((double)res->box_x4), @((double)res->box_y4)]
            ],
            @"frame": @{
                @"width": @((double)(maxX - minX)),
                @"height": @((double)(maxY - minY)),
                @"top": @((double)minY),
                @"left": @((double)minX)
            }
        };
        [resultArray addObject:dict];
    }
    return resultArray;
}

RCT_EXPORT_METHOD(initialize:(nullable NSDictionary *)configDict
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    NSError *error = nil;
    NSDictionary *config = [self validatedInitializeConfig:configDict error:&error];
    if (config == nil) {
        reject(@"INIT_ERROR", [NSString stringWithFormat:@"Invalid initialization configuration: %@", error.localizedDescription], error);
        return;
    }

    NSDictionary *models = config[@"models"] ?: @{};
    NSString *(^resolveModel)(NSString *, NSString *) = ^NSString *(NSString *key, NSString *fallback) {
        NSString *val = models[key] ?: fallback;
        if (val == nil) return nil;
        NSString *resolved = [self getResourcePath:val];
        return resolved ?: val;
    };

    NSString *preset = config[@"preset"] ?: @"ppv6";
    NSString *detModel = resolveModel(@"detection", @"det.rten");
    NSString *recModel = resolveModel(@"recognition", @"rec.rten");
    NSString *dictPath = resolveModel(@"dictionary", @"dict.txt");

    NSMutableDictionary *resolvedConfig = [NSMutableDictionary dictionaryWithDictionary:@{
        @"template": preset,
        @"detection": @{ @"modelPath": detModel ?: @"" },
        @"recognition": @{ @"modelPath": recModel ?: @"", @"dictPath": dictPath ?: @"" }
    }];

    NSString *clsPath = resolveModel(@"classification", nil);
    if (clsPath) {
        resolvedConfig[@"classification"] = @{ @"enabled": @YES, @"modelPath": clsPath };
    }
    NSString *oriPath = resolveModel(@"orientation", nil);
    if (oriPath) {
        resolvedConfig[@"orientation"] = @{ @"enabled": @YES, @"modelPath": oriPath };
    }

    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:resolvedConfig options:0 error:&error];
    if (jsonData == nil) {
        reject(@"INIT_ERROR", @"Failed to serialize configuration", error);
        return;
    }
    NSString *jsonStr = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];

    ROCRHandle replacement = rocr_initialize(jsonStr.UTF8String);
    if (replacement == NULL) {
        reject(@"INIT_ERROR", @"Failed to initialize RustO with config", nil);
        return;
    }

    if (_rustoInstance != NULL) {
        rocr_free(_rustoInstance);
    }
    _rustoInstance = replacement;
    resolve(nil);
}

RCT_EXPORT_METHOD(detectText:(NSDictionary *)source
                  options:(nullable NSDictionary *)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    if (_rustoInstance == NULL) {
        reject(@"NOT_INITIALIZED", @"RustO not initialized. Call initialize() first.", nil);
        return;
    }

    NSSet *allowedKeys = [NSSet setWithObjects:@"uri", @"base64", nil];
    if (source.count != 1) {
        reject(@"INVALID_SOURCE", @"Provide exactly one non-empty source key: uri or base64.", nil);
        return;
    }
    NSString *key = source.allKeys.firstObject;
    if (![allowedKeys containsObject:key]) {
        reject(@"INVALID_SOURCE", @"Provide exactly one non-empty source key: uri or base64.", nil);
        return;
    }
    id rawVal = source[key];
    if (![rawVal isKindOfClass:[NSString class]]) {
        reject(@"INVALID_SOURCE", @"Provide exactly one non-empty source key: uri or base64.", nil);
        return;
    }
    NSString *value = [rawVal stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (value.length == 0) {
        reject(@"INVALID_SOURCE", @"Provide exactly one non-empty source key: uri or base64.", nil);
        return;
    }

    NSError *error = nil;
    NSDictionary *runtimeOptions = [self validatedRuntimeOptions:options error:&error];
    if (runtimeOptions == nil) {
        reject(@"INVALID_OPTIONS", [NSString stringWithFormat:@"Invalid runtime options: %@", error.localizedDescription], error);
        return;
    }

    NSString *output = runtimeOptions[@"output"];
    NSData *optJsonData = [NSJSONSerialization dataWithJSONObject:runtimeOptions options:0 error:&error];
    if (optJsonData == nil) {
        reject(@"INVALID_OPTIONS", @"Invalid runtime options.", error);
        return;
    }
    NSString *optionsJson = [[NSString alloc] initWithData:optJsonData encoding:NSUTF8StringEncoding];

    if ([key isEqualToString:@"base64"]) {
        NSString *encoded = value;
        NSRange commaRange = [value rangeOfString:@"base64,"];
        if (commaRange.location != NSNotFound) {
            encoded = [value substringFromIndex:commaRange.location + commaRange.length];
        }
        NSData *data = [[NSData alloc] initWithBase64EncodedString:encoded options:NSDataBase64DecodingIgnoreUnknownCharacters];
        if (data == nil || data.length == 0) {
            reject(@"INVALID_SOURCE", @"Base64 image must decode to non-empty data.", nil);
            return;
        }

        if ([output isEqualToString:@"spatial"]) {
            char *text = rocr_detect_text_data_spatial(_rustoInstance, data.bytes, data.length, optionsJson.UTF8String);
            if (text == NULL) {
                reject(@"OCR_ERROR", @"OCR recognition failed", nil);
                return;
            }
            NSString *resStr = [NSString stringWithUTF8String:text];
            rocr_free_string(text);
            resolve(resStr);
            return;
        }

        void *resultsPtr = NULL;
        size_t count = 0;
        int status = rocr_detect_text_data(_rustoInstance, data.bytes, data.length, optionsJson.UTF8String, &resultsPtr, &count);
        if (status != 0 || resultsPtr == NULL) {
            reject(@"OCR_ERROR", [NSString stringWithFormat:@"OCR recognition failed with status: %d", status], nil);
            return;
        }
        NSArray *resArray = [self convertResultsToArray:resultsPtr count:count];
        rocr_free_results(resultsPtr, count);
        resolve(resArray);
        return;
    }

    // URI handling
    if (![value hasPrefix:@"file:"] && ![value hasPrefix:@"/"]) {
        reject(@"INVALID_SOURCE", @"Unsupported iOS URI scheme.", nil);
        return;
    }
    NSString *path = [self resolveFilePath:value];
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        reject(@"FILE_NOT_FOUND", [NSString stringWithFormat:@"Image file not found: %@", value], nil);
        return;
    }

    if ([output isEqualToString:@"spatial"]) {
        char *text = rocr_detect_text_file_spatial(_rustoInstance, path.UTF8String, optionsJson.UTF8String);
        if (text == NULL) {
            reject(@"OCR_ERROR", @"OCR recognition failed", nil);
            return;
        }
        NSString *resStr = [NSString stringWithUTF8String:text];
        rocr_free_string(text);
        resolve(resStr);
        return;
    }

    void *resultsPtr = NULL;
    size_t count = 0;
    int status = rocr_detect_text_file(_rustoInstance, path.UTF8String, optionsJson.UTF8String, &resultsPtr, &count);
    if (status != 0 || resultsPtr == NULL) {
        reject(@"OCR_ERROR", [NSString stringWithFormat:@"OCR recognition failed with status: %d", status], nil);
        return;
    }
    NSArray *resArray = [self convertResultsToArray:resultsPtr count:count];
    rocr_free_results(resultsPtr, count);
    resolve(resArray);
}

@end
