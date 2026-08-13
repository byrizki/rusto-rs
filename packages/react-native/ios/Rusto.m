#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(Rusto, NSObject)

RCT_EXTERN_METHOD(initialize:(nullable NSDictionary *)config
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(detectText:(NSDictionary *)source
                  options:(nullable NSDictionary *)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

+ (BOOL)requiresMainQueueSetup { return NO; }
@end
