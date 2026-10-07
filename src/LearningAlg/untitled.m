function double_pendulum_v2()
    % --- פרמטרים ---
    g = 9.81;
    m1 = 1.0;  L1 = 1.0;
    m2 = 1.0;  L2 = 1.0;
    
    % תנאי התחלה (קואורדינטות מוכללות: [theta1, d_theta1, theta2, d_theta2])
    y0 = [pi/2, 0, pi/4, 0]; 
    
    % זמן סימולציה: צעד זמן קטן יותר (0.01) נותן פתרון רציף יותר
    tspan = 0:0.01:20; 

    % פתרון
    [t, y] = ode45(@(t, y) pendulum_dynamics(t, y, m1, m2, L1, L2, g), tspan, y0);

    % המרה לקואורדינטות קרטזיות
    x1 = L1 * sin(y(:,1));
    y1 = -L1 * cos(y(:,1));
    x2 = x1 + L2 * sin(y(:,3));
    y2 = y1 - L2 * cos(y(:,3));

    % גרפיקה
    fig = figure('Color', 'w');
    hold on; axis equal; grid on;
    limit = (L1 + L2) * 1.1;
    axis([-limit limit -limit limit]);
    
    h_arm = plot(0, 0, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'Color', [0.2 0.2 0.2]);
    h_trail = plot(x2(1), y2(1), 'r', 'LineWidth', 1); 

    % אנימציה חלקה
    tic; % התחלת שעון למדידת זמן אמת
    for k = 2:length(t)
        % חישוב הזמן שעבר מאז תחילת הריצה כדי להתאים לזמן אמת
        while toc < t(k) * 1.5 % המכפלה ב-1.5 היא "הילוך איטי" (Slow Motion)
            % מחכה כדי להתאים לקצב הרצוי
        end
        
        set(h_arm, 'XData', [0, x1(k), x2(k)], 'YData', [0, y1(k), y2(k)]);
        set(h_trail, 'XData', x2(max(1, k-100):k), 'YData', y2(max(1, k-100):k)); % שובל קצר ודינמי
        
        drawnow limitrate; % מונע עומס על המעבד ומציג בצורה רציפה
        if ~ishandle(fig), break; end % עצירה אם סגרת את החלון
    end
end

function dydt = pendulum_dynamics(~, y, m1, m2, L1, L2, g)
    % פירוק וקטור המצב לקואורדינטות מוכללות
    theta = y(1);  % theta 1
    theta  = y(2);  % theta 1 dot
    t2 = y(3);  % theta 2
    w2 = y(4);  % theta 2 dot
    
    % --- כאן אתה מזין את המשוואות שלך ---
    % לדוגמה, נשתמש במשוואות התנועה הסטנדרטיות:
    delta = t2 - t1;
    den = (2*m1 + m2 - m2*cos(2*t1 - 2*t2));
    
    % תאוצות זוויתיות (theta double dot)
    dw1 = (-g*(2*m1 + m2)*sin(t1) - m2*g*sin(t1 - 2*t2) - ...
           2*sin(delta)*m2*(w2^2*L2 + w1^2*L1*cos(delta))) / (L1 * den);
       
    dw2 = (2*sin(delta)*(w1^2*L1*(m1 + m2) + g*(m1 + m2)*cos(t1) + ...
           w2^2*L2*m2*cos(delta))) / (L2 * den);

    % בניית הנגזרת של וקטור המצב: [q_dot; q_double_dot]
    dydt = [w1; dw1; w2; dw2];
end