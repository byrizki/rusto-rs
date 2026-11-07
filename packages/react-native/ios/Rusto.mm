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
    int region_id;
    int layout_type;
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
const char *rocr_version(void);
int rocr_is_ready(ROCRHandle handle);


#ifdef __cplusplus
}
#endif

static NSString *layoutTypeNameFromCode(int code) {
    switch (code) {
        case 0: return @"text";
        case 1: return @"title";
        case 2: return @"figure";
        case 3: return @"figureCaption";
        case 4: return @"table";
        case 5: return @"tableCaption";
        case 6: return @"header";
        case 7: return @"footer";
        case 8: return @"reference";
        case 9: return @"equation";
        default: return nil;
    }
}

static BOOL isJsonBoolean(id value) {
    return [value isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

static BOOL isJsonNumber(id value) {
    return [value isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID();
}

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
    for (NSString *candidate in @[@"rten", @"onnx", @"mnn", @"txt", @"json"]) {
        if (![candidateExts containsObject:candidate]) {
            [candidateExts addObject:candidate];
        }
    }

    NSArray<NSString *> *bundleNames = @[
        @"RustOModels", @"RustoModels",
        @"RustOModels_Indra",
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

    NSSet *allowed = [NSSet setWithObjects:@"preset", @"model", @"modelPath", @"lang",
                                           @"cacheDir", @"enableReadingOrder", @"layoutVariant",
                                           @"recBatchNum", @"models", nil];
    NSSet *keys = [NSSet setWithArray:config.allKeys];
    if (![keys isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"unknown key");
        return nil;
    }

    if (config[@"preset"] != nil) {
        id preset = config[@"preset"];
        if (![preset isKindOfClass:[NSString class]] ||
            ![@[@"indra", @"ppv6", @"ppv5", @"ppv4", @"ppv3"] containsObject:preset]) {
            if (error) *error = makeValidationError(@"preset is invalid");
            return nil;
        }
    }

    for (NSString *strKey in @[@"model", @"modelPath", @"lang", @"cacheDir"]) {
        if (config[strKey] != nil) {
            id val = config[strKey];
            if (![val isKindOfClass:[NSString class]] ||
                [[val stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] length] == 0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ must be a non-empty string", strKey]);
                return nil;
            }
        }
    }

    if (config[@"enableReadingOrder"] != nil && !isJsonBoolean(config[@"enableReadingOrder"])) {
        if (error) *error = makeValidationError(@"enableReadingOrder must be a boolean");
        return nil;
    }

    if (config[@"layoutVariant"] != nil) {
        id variant = config[@"layoutVariant"];
        if (![variant isKindOfClass:[NSString class]] ||
            (![variant isEqualToString:@"cdla"] && ![variant isEqualToString:@"publaynet"])) {
            if (error) *error = makeValidationError(@"layoutVariant is invalid");
            return nil;
        }
    }

    if (config[@"recBatchNum"] != nil) {
        id val = config[@"recBatchNum"];
        if (!isJsonNumber(val)) {
            if (error) *error = makeValidationError(@"recBatchNum is invalid");
            return nil;
        }
        double d = [val doubleValue];
        if (!isfinite(d) || d < 1.0 || floor(d) != d) {
            if (error) *error = makeValidationError(@"recBatchNum is invalid");
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
        NSSet *allowedModels = [NSSet setWithObjects:@"detection", @"recognition", @"dictionary",
                                                     @"classification", @"orientation", @"unwarp", @"layout", nil];
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

- (NSDictionary *)validatedPreDetection:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"preDetection must be an object");
        return nil;
    }
    NSDictionary *pre = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"descreenStrength", @"targetMaxSide", @"cropPaddingX", @"cropPaddingY", nil];
    if (![[NSSet setWithArray:pre.allKeys] isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"preDetection contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"descreenStrength", @"cropPaddingX", @"cropPaddingY"]) {
        if (pre[key] != nil) {
            id val = pre[key];
            if (!isJsonNumber(val) || !isfinite([val doubleValue]) || [val doubleValue] < 0.0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
            result[key] = val;
        }
    }
    if (pre[@"targetMaxSide"] != nil) {
        id val = pre[@"targetMaxSide"];
        double d = isJsonNumber(val) ? [val doubleValue] : -1.0;
        if (!isfinite(d) || d < 1.0 || floor(d) != d) {
            if (error) *error = makeValidationError(@"targetMaxSide is invalid");
            return nil;
        }
        result[@"targetMaxSide"] = val;
    }
    return result;
}

- (NSDictionary *)validatedRuntimeDetection:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"detection must be an object");
        return nil;
    }
    NSDictionary *detection = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"limitSideLen", @"limitType", @"mean", @"std", nil];
    if (![[NSSet setWithArray:detection.allKeys] isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"detection contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if (detection[@"limitSideLen"] != nil) {
        id rawVal = detection[@"limitSideLen"];
        double val = isJsonNumber(rawVal) ? [rawVal doubleValue] : -1.0;
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
                if (!isJsonNumber(item)) {
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

- (NSDictionary *)validatedPostDetection:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"postDetection must be an object");
        return nil;
    }
    NSDictionary *post = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"threshold", @"boxThreshold", @"unclipRatio", @"useDilation",
                                           @"scoreMode", @"maxCandidates", @"minHeight", @"minSideLen",
                                           @"maxSideLen", @"widthHeightRatio", nil];
    if (![[NSSet setWithArray:post.allKeys] isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"postDetection contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"threshold", @"boxThreshold"]) {
        if (post[key] != nil) {
            id val = post[key];
            double d = isJsonNumber(val) ? [val doubleValue] : -1.0;
            if (!isfinite(d) || d < 0.0 || d > 1.0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
            result[key] = val;
        }
    }
    if (post[@"maxCandidates"] != nil) {
        id val = post[@"maxCandidates"];
        double d = isJsonNumber(val) ? [val doubleValue] : -1.0;
        if (!isfinite(d) || d < 1.0 || floor(d) != d) {
            if (error) *error = makeValidationError(@"maxCandidates is invalid");
            return nil;
        }
        result[@"maxCandidates"] = val;
    }
    for (NSString *key in @[@"unclipRatio", @"minHeight", @"minSideLen", @"maxSideLen"]) {
        if (post[key] != nil) {
            id val = post[key];
            double d = isJsonNumber(val) ? [val doubleValue] : -1.0;
            if (!isfinite(d) || d <= 0.0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
            result[key] = val;
        }
    }
    if (post[@"minSideLen"] != nil && post[@"maxSideLen"] != nil &&
        [post[@"minSideLen"] doubleValue] > [post[@"maxSideLen"] doubleValue]) {
        if (error) *error = makeValidationError(@"minSideLen must be <= maxSideLen");
        return nil;
    }
    if (post[@"widthHeightRatio"] != nil) {
        id val = post[@"widthHeightRatio"];
        double d = isJsonNumber(val) ? [val doubleValue] : 0.0;
        if (!isfinite(d) || (d <= 0.0 && d != -1.0)) {
            if (error) *error = makeValidationError(@"widthHeightRatio is invalid");
            return nil;
        }
        result[@"widthHeightRatio"] = val;
    }
    if (post[@"useDilation"] != nil) {
        if (!isJsonBoolean(post[@"useDilation"])) {
            if (error) *error = makeValidationError(@"useDilation must be a boolean");
            return nil;
        }
        result[@"useDilation"] = post[@"useDilation"];
    }
    if (post[@"scoreMode"] != nil) {
        id val = post[@"scoreMode"];
        if (![val isKindOfClass:[NSString class]] || (![val isEqualToString:@"fast"] && ![val isEqualToString:@"slow"])) {
            if (error) *error = makeValidationError(@"scoreMode is invalid");
            return nil;
        }
        result[@"scoreMode"] = val;
    }
    return result;
}

- (NSDictionary *)validatedPreRecognition:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"preRecognition must be an object");
        return nil;
    }
    NSDictionary *pre = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"lineYThreshold", @"wordXThreshold", nil];
    if (![[NSSet setWithArray:pre.allKeys] isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"preRecognition contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSString *key in @[@"lineYThreshold", @"wordXThreshold"]) {
        if (pre[key] != nil) {
            id val = pre[key];
            double d = isJsonNumber(val) ? [val doubleValue] : -1.0;
            if (!isfinite(d) || d < 0.0) {
                if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ is invalid", key]);
                return nil;
            }
            result[key] = val;
        }
    }
    return result;
}

- (NSDictionary *)validatedRecognition:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"recognition must be an object");
        return nil;
    }
    NSDictionary *rec = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"batchSize", nil];
    if (![[NSSet setWithArray:rec.allKeys] isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"recognition contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if (rec[@"batchSize"] != nil) {
        id val = rec[@"batchSize"];
        double d = isJsonNumber(val) ? [val doubleValue] : -1.0;
        if (!isfinite(d) || d < 1.0 || floor(d) != d) {
            if (error) *error = makeValidationError(@"batchSize is invalid");
            return nil;
        }
        result[@"batchSize"] = val;
    }
    return result;
}

- (NSDictionary *)validatedPostRecognition:(id)raw error:(NSError **)error {
    if (![raw isKindOfClass:[NSDictionary class]]) {
        if (error) *error = makeValidationError(@"postRecognition must be an object");
        return nil;
    }
    NSDictionary *post = (NSDictionary *)raw;
    NSSet *allowed = [NSSet setWithObjects:@"scoreThreshold", nil];
    if (![[NSSet setWithArray:post.allKeys] isSubsetOfSet:allowed]) {
        if (error) *error = makeValidationError(@"postRecognition contains an unknown key");
        return nil;
    }
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if (post[@"scoreThreshold"] != nil) {
        id val = post[@"scoreThreshold"];
        double d = isJsonNumber(val) ? [val doubleValue] : -1.0;
        if (!isfinite(d) || d < 0.0 || d > 1.0) {
            if (error) *error = makeValidationError(@"scoreThreshold is invalid");
            return nil;
        }
        result[@"scoreThreshold"] = val;
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
    NSSet *allowed = [NSSet setWithObjects:@"output", @"autoUnwarp", @"layoutReading",
                                          @"orientationDetection", @"lineClassification",
                                          @"preDetection", @"detection", @"postDetection",
                                          @"preRecognition", @"recognition", @"postRecognition", nil];
    if (![[NSSet setWithArray:options.allKeys] isSubsetOfSet:allowed]) {
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

    for (NSString *key in @[@"autoUnwarp", @"layoutReading", @"orientationDetection", @"lineClassification"]) {
        if (options[key] != nil && !isJsonBoolean(options[key])) {
            if (error) *error = makeValidationError([NSString stringWithFormat:@"%@ must be a boolean", key]);
            return nil;
        }
    }

    if (options[@"preDetection"] != nil) {
        NSDictionary *sub = [self validatedPreDetection:options[@"preDetection"] error:error];
        if (sub == nil) return nil;
        options[@"preDetection"] = sub;
    }
    if (options[@"detection"] != nil) {
        NSDictionary *sub = [self validatedRuntimeDetection:options[@"detection"] error:error];
        if (sub == nil) return nil;
        options[@"detection"] = sub;
    }
    if (options[@"postDetection"] != nil) {
        NSDictionary *sub = [self validatedPostDetection:options[@"postDetection"] error:error];
        if (sub == nil) return nil;
        options[@"postDetection"] = sub;
    }
    if (options[@"preRecognition"] != nil) {
        NSDictionary *sub = [self validatedPreRecognition:options[@"preRecognition"] error:error];
        if (sub == nil) return nil;
        options[@"preRecognition"] = sub;
    }
    if (options[@"recognition"] != nil) {
        NSDictionary *sub = [self validatedRecognition:options[@"recognition"] error:error];
        if (sub == nil) return nil;
        options[@"recognition"] = sub;
    }
    if (options[@"postRecognition"] != nil) {
        NSDictionary *sub = [self validatedPostRecognition:options[@"postRecognition"] error:error];
        if (sub == nil) return nil;
        options[@"postRecognition"] = sub;
    }

    return options;
}

- (NSArray<NSDictionary *> *)convertResultsToArray:(void *)resultsPtr count:(size_t)count {
    NSMutableArray<NSDictionary *> *resultArray = [NSMutableArray arrayWithCapacity:count];
    CTextResult *results = (CTextResult *)resultsPtr;

    for (size_t i = 0; i < count; i++) {
        CTextResult *res = &results[i];
        NSString *textString = res->text ? [NSString stringWithUTF8String:res->text] : @"";

        NSMutableDictionary *dict = [NSMutableDictionary dictionaryWithDictionary:@{
            @"text": textString,
            @"score": @((double)res->score),
            @"box_points": @[
                @[@((double)res->box_x1), @((double)res->box_y1)],
                @[@((double)res->box_x2), @((double)res->box_y2)],
                @[@((double)res->box_x3), @((double)res->box_y3)],
                @[@((double)res->box_x4), @((double)res->box_y4)]
            ],
            @"frame": @{
                @"width": @((double)res->frame_width),
                @"height": @((double)res->frame_height),
                @"top": @((double)res->frame_top),
                @"left": @((double)res->frame_left)
            }
        }];
        if (res->region_id >= 0) {
            dict[@"region_id"] = @(res->region_id);
        }
        NSString *layoutName = layoutTypeNameFromCode(res->layout_type);
        if (layoutName != nil) {
            dict[@"layout_type"] = layoutName;
        }
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
    NSString *(^resolveModel)(NSString *) = ^NSString *(NSString *key) {
        NSString *val = models[key];
        if (val == nil) return nil;
        NSString *resolved = [self getResourcePath:val];
        return resolved ?: val;
    };

    NSMutableDictionary *resolvedConfig = [NSMutableDictionary dictionary];
    if (config[@"preset"] != nil) resolvedConfig[@"template"] = config[@"preset"];
    if (config[@"model"] != nil) resolvedConfig[@"model"] = config[@"model"];
    if (config[@"modelPath"] != nil) resolvedConfig[@"modelPath"] = config[@"modelPath"];
    if (config[@"lang"] != nil) resolvedConfig[@"lang"] = config[@"lang"];
    if (config[@"enableReadingOrder"] != nil) resolvedConfig[@"enableReadingOrder"] = config[@"enableReadingOrder"];
    if (config[@"layoutVariant"] != nil) resolvedConfig[@"layoutVariant"] = config[@"layoutVariant"];

    NSString *cacheDir = config[@"cacheDir"];
    if (cacheDir == nil) {
        NSArray *cachePaths = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
        if (cachePaths.count > 0) {
            cacheDir = [cachePaths[0] stringByAppendingPathComponent:@"rusto/models"];
        }
    }
    if (cacheDir != nil) {
        resolvedConfig[@"cacheDir"] = cacheDir;
    }

    NSString *detModel = resolveModel(@"detection");
    NSString *recModel = resolveModel(@"recognition");
    NSString *dictPath = resolveModel(@"dictionary");

    if (detModel == nil && recModel == nil && config[@"model"] == nil && config[@"modelPath"] == nil) {
        NSString *bundledDet = [self getResourcePath:@"det.mnn"] ?: [self getResourcePath:@"det.rten"];
        NSString *bundledRec = [self getResourcePath:@"rec.mnn"] ?: [self getResourcePath:@"rec.rten"];
        NSString *bundledDict = [self getResourcePath:@"meta.json"] ?: [self getResourcePath:@"dict.txt"];
        if (bundledDet != nil && bundledRec != nil && bundledDict != nil) {
            detModel = bundledDet;
            recModel = bundledRec;
            dictPath = bundledDict;
        }
    }

    if (detModel != nil) {
        resolvedConfig[@"detection"] = @{ @"modelPath": detModel };
    }
    if (recModel != nil || dictPath != nil || config[@"recBatchNum"] != nil) {
        NSMutableDictionary *recDict = [NSMutableDictionary dictionary];
        if (recModel != nil) recDict[@"modelPath"] = recModel;
        if (dictPath != nil) recDict[@"dictPath"] = dictPath;
        if (config[@"recBatchNum"] != nil) recDict[@"recBatchNum"] = config[@"recBatchNum"];
        resolvedConfig[@"recognition"] = recDict;
    }

    NSString *clsPath = resolveModel(@"classification");
    if (clsPath) resolvedConfig[@"classification"] = @{ @"enabled": @YES, @"modelPath": clsPath };
    NSString *oriPath = resolveModel(@"orientation");
    if (oriPath) resolvedConfig[@"orientation"] = @{ @"enabled": @YES, @"modelPath": oriPath };
    NSString *unwarpPath = resolveModel(@"unwarp");
    if (unwarpPath) resolvedConfig[@"unwarp"] = @{ @"enabled": @YES, @"modelPath": unwarpPath };
    NSString *layoutPath = resolveModel(@"layout");
    if (layoutPath) resolvedConfig[@"layout"] = @{ @"enabled": @YES, @"modelPath": layoutPath };

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

RCT_EXPORT_METHOD(close:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    if (_rustoInstance != NULL) {
        rocr_free(_rustoInstance);
        _rustoInstance = NULL;
    }
    resolve(nil);
}

RCT_EXPORT_METHOD(isReady:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    BOOL ready = (_rustoInstance != NULL && rocr_is_ready(_rustoInstance) != 0);
    resolve(@(ready));
}

RCT_EXPORT_METHOD(version:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
    const char *v = rocr_version();
    resolve(v != NULL ? [NSString stringWithUTF8String:v] : @"0.4.0");
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
