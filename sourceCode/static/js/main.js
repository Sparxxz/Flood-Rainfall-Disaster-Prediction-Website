jQuery(document).on('ready', function ($) {
    "use strict";

    /*---------------------------
        SEARCH BOX
    ----------------------------*/
    jQuery('.search-box').on('click', function () {
        jQuery('.search-form').slideToggle();
    });


    /*--------------------------
        STICKY MAINMENU
    ---------------------------*/
    $("#mainmenu-area").sticky({
        topSpacing: 0
    });


    /*---------------------------
        SMOOTH SCROLL
    -----------------------------*/
    $('ul#nav li a[href^="#"], a.navbar-brand, a.scrolltotop').on('click', function (event) {
        var id = $(this).attr("href");
        var offset = 60;
        var target = $(id).offset().top - offset;
        $('html, body').animate({
            scrollTop: target
        }, 1500, "easeInOutExpo");
        event.preventDefault();
    });


    /*----------------------------
        MOBILE & DROPDOWN MENU
    ------------------------------*/
    jQuery('.stellarnav').stellarNav({
        theme: 'dark'
    });

    /*----------------------------
        SCROLL TO TOP
    ------------------------------*/
    $(window).scroll(function () {
        var totalHeight = $(window).scrollTop();
        if (totalHeight > 300) {
            $(".scrolltotop").fadeIn();
        } else {
            $(".scrolltotop").fadeOut();
        }
    });


    /*--------------------------
       HOME PARALLAX BACKGROUND
    ----------------------------*/
    $(window).stellar({
        responsive: true,
        positionProperty: 'position',
        horizontalScrolling: false
    });


    /*---------------------------
        HOME SLIDER
    -----------------------------*/
    var $homeSlider = $('.welcome-slider-area');
    $homeSlider.owlCarousel({
        merge: true,
        smartSpeed: 2000,
        loop: true,
        nav: true,
        navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
        autoplay: true,
        autoplayTimeout: 5000,
        margin: 0,
        animateIn: 'fadeIn',
        animateOut: 'fadeOut',
        responsiveClass: true,
        responsive: {
            0: {
                items: 1
            },
            600: {
                items: 1
            },
            1000: {
                items: 1
            },
            1200: {
                items: 1
            }
        }
    });


    /*------------------------------
        TESTMONIAL SLIDER
    -------------------------------*/
    var client_photo2 = $('.client_details');
    client_photo2.owlCarousel({
        loop: true,
        margin: 30,
        autoplay: false,
        dots: true,
        autoplayTimeout: 4000,
        smartSpeed: 600,
        responsive: {
            0: {
                items: 1
            },
            600: {
                items: 1
            },
            992: {
                items: 1
            }
        }
    });
    
    var client_photo = $('.client_photo');
    client_photo.owlCarousel({
        loop: true,
        center: true,
        margin: 0,
        dots: true,
        autoplayTimeout: 4000,
        smartSpeed: 600,
        mouseDrag: true,
        touchDrag: false,
        center: true,
        responsive: {
            0: {
                items: 1
            },
            600: {
                items: 3
            },
            992: {
                items: 3
            }
        }
    });
    $('.client_nav .testi_next').on('click', function () {
        client_photo.trigger('next.owl.carousel');
    });
    $('.client_nav .testi_prev').on('click', function () {
        client_photo.trigger('prev.owl.carousel');
    });

    client_photo.on('translate.owl.carousel', function (property) {
        $('.client-details-content .owl-dot:eq(' + property.page.index + ')').click();
    });
    client_photo2.on('translate.owl.carousel', function (property) {
        $('.client-photo-list .owl-dot:eq(' + property.page.index + ')').click();
    });

    /*----------------------------
        TAB PANEL ACTIVE
    ------------------------------*/
    $('.panel').on('click', function (e) {
        $('.panel').removeClass('active');
        var $this = $(this);
        if (!$this.hasClass('active')) {
            $this.addClass('active');
        }
        e.preventDefault();
    });


    /*---------------------------
        TESTMONIAL SLIDER
    -----------------------------*/
    $('.client-list').owlCarousel({
        merge: true,
        smartSpeed: 1000,
        loop: true,
        nav: false,
        navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
        autoplay: false,
        autoplayTimeout: 2000,
        margin: 10,
        responsiveClass: true,
        responsive: {
            0: {
                items: 2
            },
            600: {
                items: 4
            },
            1000: {
                items: 6
            }
        }
    });


    /*--------------------------
        ACTIVE WOW JS
    ----------------------------*/
    new WOW().init();


}(jQuery));



jQuery(window).on('load', function () {
    "use strict";
    /*--------------------------
        PRE LOADER
    ----------------------------*/
    $(".preeloader").fadeOut(1000);

});
