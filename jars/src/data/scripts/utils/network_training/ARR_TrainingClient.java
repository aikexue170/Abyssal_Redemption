package data.scripts.utils.network_training;

import com.fs.starfarer.api.combat.ShipAPI;
import data.scripts.utils.network.SimpleSocketClient;

/**
 * 与 Python 训练端通信的客户端（本 Mod 唯一的网络通道）。
 *
 * 协议（TCP，行文本，帧以 END 结尾）：
 *   Java -> Python:
 *       STATE,<simTime>,<shipId>,<x>,<y>,<vx>,<vy>,<facing>,<angVel>
 *       END
 *   Python -> Java:
 *       ACT,<shipId>,<move>,<turn>,<strafe>   三轴开度，范围 [-1,1]
 *       RESET,<shipId|ALL>                    复位到出生点并清零速度/角速度
 *       END
 */
public class ARR_TrainingClient {

    public static final String SERVER_IP = "127.0.0.1";
    public static final int SERVER_PORT = 65432;

    private final SimpleSocketClient client = new SimpleSocketClient(SERVER_IP, SERVER_PORT);

    public boolean connect() {
        return client.connect();
    }

    public boolean isConnected() {
        return client.isConnected();
    }

    public void disconnect() {
        client.disconnect();
    }

    /** 发送一条舰船状态行。 */
    public void sendShipState(ShipAPI ship, float simTime) {
        if (!client.isConnected() || ship == null) return;
        client.send("STATE," + simTime + "," + ship.getId() + ","
                + ship.getLocation().x + "," + ship.getLocation().y + ","
                + ship.getVelocity().x + "," + ship.getVelocity().y + ","
                + ship.getFacing() + "," + ship.getAngularVelocity());
    }

    /** 结束一帧状态发送。 */
    public void endStateFrame() {
        if (client.isConnected()) {
            client.send("END");
        }
    }

    /** 一条来自 Python 端的消息。 */
    public static class Message {
        public enum Type { ACT, RESET }

        public Type type;
        public String shipId;   // RESET 时为 "ALL" 表示全部舰船
        public float move;
        public float turn;
        public float strafe;
    }

    /**
     * 取出下一条已解析消息；没有则返回 null。
     * 无法识别的行会被丢弃。
     */
    public Message nextMessage() {
        while (client.hasMessage()) {
            String line = client.getMessage();
            if (line == null) return null;
            line = line.trim();
            if (line.isEmpty() || line.equals("END")) continue;

            String[] p = line.split(",");
            try {
                if (p.length == 5 && p[0].equalsIgnoreCase("ACT")) {
                    Message m = new Message();
                    m.type = Message.Type.ACT;
                    m.shipId = p[1];
                    m.move = clamp(Float.parseFloat(p[2]));
                    m.turn = clamp(Float.parseFloat(p[3]));
                    m.strafe = clamp(Float.parseFloat(p[4]));
                    return m;
                }
                if (p.length == 2 && p[0].equalsIgnoreCase("RESET")) {
                    Message m = new Message();
                    m.type = Message.Type.RESET;
                    m.shipId = p[1];
                    return m;
                }
            } catch (NumberFormatException ignored) {
                // 丢弃格式错误的行
            }
        }
        return null;
    }

    private static float clamp(float v) {
        return Math.max(-1f, Math.min(1f, v));
    }
}
