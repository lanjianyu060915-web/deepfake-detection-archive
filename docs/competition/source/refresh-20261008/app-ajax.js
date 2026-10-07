/**
 * ajax的工具类 -- rest接口
 * @author zjingcan@linewell.com
 * @since 2017-03-01
 */
;define("app-ajax", function(require, exports, module) {
	var ajaxUtils = require("ajax-utils-rest");
	var cookies = require("common/js/utils/cookie");
	var loadingUtils = require("common/js/common/loading-utils");
	var notify = require("notify");
	var constants = require("constants");
	var servicesConfig = require("services-config");
	var loadingButtons = [];
	
	/* 基础通信参数  */
	var _authClient = function(forumToken) {
		
		var deviceId = cookies.getCookie("deviceId") || "PC";
		var auth = {
			authParams : {
				timestamp : new Date().getTime(),
				token : forumToken ? cookies.getCookie("token_ai_forum") : cookies.getCookie("token_ai"),
				deviceId : deviceId
			},
			clientParams : {
				os: "PC",
				network: "",
				deviceId: deviceId,
				appVersion: ""
			},
			appId : INNO_CHINA_CONSTANTS.APP_ID || "",
			siteId : INNO_CHINA_CONSTANTS.SITE_ID || "",
			siteAreaCode : INNO_CHINA_CONSTANTS.SITE_AREA_CODE || "",
		};
		
		return auth;
	};
	
	/**
	 * 提交表单数据
	 * @param {Object} params
	 */
	var _postJson = function(params){
		
		var $loadingButtons = $(loadingButtons);
		
		// 判断按钮状态
		if ($loadingButtons.filter(".loading-button,.disabled").length > 0) {
			return;
		}
		$loadingButtons.addClass("loading-button");
		var authClient =  _authClient(params.forumToken);
		
		// 默认参数
		var defaultParams = {
			service : "",            // 服务的配置名称
			data : authClient, // 发送的data
			success : function(d){}, // 成功后回调
			error : function(d){},   // 失败后回调
			autoShowWait : false,   // 自动显示菊花
			loadingText : "正在加载", // 加载的提示语
			autoCloseWait : true,  // 自动关闭菊花
			headers : {}, // 额外参数
			isAsync : true
		};
		var ajaxParams = $.extend(true, {}, defaultParams, params);
		
		// rest请求路径
		ajaxParams["url"] = constants["SERVICE_URL"] + servicesConfig[ajaxParams["service"]];
		
		// 增加请求头部
		ajaxParams["beforeSend"] = function (request) {
			request.setRequestHeader("deviceId", authClient.authParams.deviceId);
		};
		
		//成功回调方法重载
		ajaxParams.success = function(d){
			try{
				if(d){
					switch(d.status){
						case 1:// 成功
							if(d.content){
								if(params.success){
									params.success.call(this, d.content, d);	
								}
							}else{
								if(params.success){
									params.success.call(this, null, d);
								}		   		
							}	 		
							break;
						case 2:// 回话过期或者未登录
							
//							// 2016-09-05 add by sxiangyu
//							// 清空缓存信息 
//							if (appSession.isLogin()) {
//								// 异步引用，不然脚本互相引用会报错
//								require.async("app-user", function(appUser) {
//									appUser.cleanLoginInfo();
//								});
//							}
//						
//							// 2016-05-13 del by xhuatang
//							// ajax交互不需要所有的都做会话过期的判断
//							if(typeof(ajaxParams["checkLogin"]) !== "undefined" && ajaxParams["checkLogin"]){
//								appPage.openLogin();
//							}
							
							break;
						default:// 失败或者其他
//							mui.toast("服务器通信异常");
							var message = d.message ? d.message : "有点忙开个小差，稍后再试~";
							if(params.error){
								params.error.call(this, message, d);
							}else{
								notify.showTip(message, "error");
							}
							break;
					}
				}
			}catch(e){
				// handle the exception
				ajaxParams.autoCloseWait && loadingUtils.close();
				console.log(e);
			}
		};
		
		// 是否显示菊花
		ajaxParams["autoShowWait"] && loadingUtils.open(ajaxParams["loadingText"]);
		
		var errorFn = ajaxParams.error;
		ajaxParams.error = function(d){
			ajaxParams.autoCloseWait && loadingUtils.close();
			if(errorFn){
				errorFn(d.message, d);
			} else if (d) {
				notify.showTip(d.message, "error");
			}
		};
		
		var complentFn = ajaxParams.complete;
		
		ajaxParams.complete = function(XMLHttpRequest,status){
			
			// 去除加载状态
			$loadingButtons.removeClass("loading-button");
			ajaxParams.autoCloseWait && loadingUtils.close();
			if(status === 'timeout' && !complentFn){//超时,status还有success,error等值的情况
				notify.showTip("网络连接超时,请稍后再试", "error");
			}
			if(complentFn){
				complentFn(XMLHttpRequest,status);
			}
		};
		
		try {
			ajaxUtils.postJson(ajaxParams).always(function () {
				
				// 去除加载状态
				$loadingButtons.removeClass("loading-button");
				ajaxParams.autoCloseWait && loadingUtils.close();
				ajaxParams.nativeWait && plus.nativeUI.closeWaiting();
			});
		} catch (e) {
			
			// 去除加载状态
			$loadingButtons.removeClass("loading-button");
			ajaxParams.autoCloseWait && loadingUtils.close();
			ajaxParams.nativeWait && plus.nativeUI.closeWaiting();
		}
		
		loadingButtons = [];
		
	};
	
	var exportsMethods = {
		/**
		 * 提交Json对象
		 * @param {Object} params 配置定义的key
		 */
		postJson:function(params){
			_postJson(params);
		},
		
		/**
		 * 设置按钮加载样式
		 * @param $button 按钮
		 */
		loadingButton : function ($button) {
			var that = this;
			that.loadingButtons($button);
			return that;
		},
		
		loadingButtons : function($button) {
			if (!$button || $button.length == 0) {
				return;
			}
			loadingButtons.push($button[0]);
		}
	};

	module.exports = exportsMethods;
});
